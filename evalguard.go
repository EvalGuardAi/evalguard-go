// Package evalguard provides a Go client for the EvalGuard API.
//
// Usage:
//
//	client, err := evalguard.NewClient("your-api-key",
//		evalguard.WithBaseURL("https://evalguard.ai/api/v1"),
//		evalguard.WithTimeout(30*time.Second),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	started, err := client.RunEval(ctx, &evalguard.RunEvalRequest{
//		Name:      "regression-suite",
//		ProjectID: "proj_abc123",
//		Model:     "gpt-4o",
//		Prompt:    "Answer concisely: {{input}}",
//		Cases:     []evalguard.EvalCase{{Input: "2+2?", ExpectedOutput: "4"}},
//		Scorers:   []string{"exact-match"},
//	})
//	// RunEval is async; poll client.GetEval(ctx, started.ID) for results.
package evalguard

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	mrand "math/rand"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxRetries     = 3
	baseRetryDelay = 500 * time.Millisecond
	// maxRetryDelay is a HARD ceiling on any single retry sleep.
	//
	// AUDIT 2026-07-25 (availability): doRaw used to sleep for
	// rateLimitErr.RetryAfter verbatim, and RetryAfter was taken straight from
	// the server's Retry-After header with no upper bound (defaulting to 60s
	// when absent). A hostile or merely mis-set `Retry-After: 3600` from an
	// intermediary WAF/CDN under load shedding parked the caller for an hour
	// per attempt — up to 2 hours on one logical call — while
	// c.httpClient.Timeout bounded only each exchange, never the sleep. A Go
	// service calling CheckFirewall inline on its request path with
	// context.Background() piled up goroutines until it fell over. Java already
	// clamps this (EvalGuardClient MAX_BACKOFF); Go was never migrated.
	maxRetryDelay = 10 * time.Second
	// retryAfterFallback is used when the server sends no parseable
	// Retry-After. Jittered by jitteredDelay, like every other branch.
	retryAfterFallback = 5 * time.Second
)

// clampRetryDelay bounds any retry sleep to maxRetryDelay. A server-supplied
// Retry-After may SHORTEN a wait; it may never extend it past the ceiling.
// Negative/zero values collapse to the exponential base so a bogus header can
// neither hang nor busy-loop the caller.
func clampRetryDelay(d time.Duration) time.Duration {
	if d <= 0 {
		return baseRetryDelay
	}
	if d > maxRetryDelay {
		return maxRetryDelay
	}
	return d
}

// jitteredDelay applies +-50% jitter to a (already clamped) delay so N clients
// that hit the same 429/5xx do not wake together and re-stampede the origin.
// math/rand is fine here: this is scheduling noise, not a secret.
func jitteredDelay(d time.Duration) time.Duration {
	d = clampRetryDelay(d)
	return time.Duration(float64(d) * (0.5 + mrand.Float64()*0.5))
}

const (
	DefaultBaseURL = "https://evalguard.ai/api/v1"
	DefaultTimeout = 30 * time.Second
	// clientVersion is the SINGLE source of truth for this SDK's version. It is
	// sent as x-evalguard-client-version on every request so an org that pins
	// allowed client versions can enforce its policy on this SDK (deep-audit
	// 2026-06-21). Bump this ONE constant per release and tag the module to
	// match — published go-sdk-v1.4.2, so the next release is go-sdk-v1.5.0.
	//
	// 2026-08-03: was 1.4.3 (an unreleased PATCH). MINOR, not PATCH, because
	// this tree now carries a deliberate BEHAVIOUR change since 1.4.2:
	// CheckFirewall / CheckFirewallAdvanced / AuditMcpServer /
	// RunAgentExecRedTeam / ScanRAGInjection return ErrCodeIndeterminate
	// instead of a zero-valued struct when the 2xx carried no verdict, plus new
	// public API (HasVerdict on four result types). Reusing a number that is
	// already in the wild under different behaviour is exactly what hid the
	// published-vs-repo drift in the Java SDK at 1.0.8.
	//
	// 2026-08-06: 1.5.0 -> 1.6.0. MINOR again, and for the same two reasons:
	// thirteen more methods refuse an unreadable verdict instead of returning a
	// zero-valued result (BEHAVIOUR), and the release is otherwise purely
	// ADDITIVE — new result types with HasVerdict, SecurityScanRequest.Depth /
	// StrategyIDs, SecurityScanResult.Mode / StatusURL / ExecutedTests /
	// ErroredTests, SecurityScanResult.Queued(). No signature changed, so no
	// consumer's build breaks. v1.5.0 is live on proxy.golang.org, so this
	// number must not be reused.
	clientVersion = "1.6.0"
	// userAgent is DERIVED from clientVersion (constant string concatenation is
	// evaluated at compile time, so this stays a plain const) so the two can
	// never drift. A prior audit found a hardcoded "evalguard-go/1.2.0" literal
	// here while clientVersion said 1.4.0 — deriving it deletes the second magic
	// string that made that regression possible.
	userAgent = "evalguard-go/" + clientVersion
)

// ErrorCode represents categorized API error codes.
type ErrorCode string

const (
	ErrCodeUnauthorized   ErrorCode = "UNAUTHORIZED"
	ErrCodeForbidden      ErrorCode = "FORBIDDEN"
	ErrCodeNotFound       ErrorCode = "NOT_FOUND"
	ErrCodeRateLimit      ErrorCode = "RATE_LIMITED"
	ErrCodeValidation     ErrorCode = "VALIDATION_ERROR"
	ErrCodeInternal       ErrorCode = "INTERNAL_ERROR"
	ErrCodeTimeout        ErrorCode = "TIMEOUT"
	ErrCodeNetworkFailure ErrorCode = "NETWORK_FAILURE"
	// ErrCodeIndeterminate is returned when the server answered 2xx but the
	// body carried NO decision field, so the security verdict could not be
	// read. It is deliberately distinct from every other code: an
	// indeterminate verdict is neither "allowed" nor a transport failure, and
	// it must never be collapsed into either.
	//
	// AUDIT 2026-08-03 (fail-open sweep, CLASS 1). Go's zero value is the same
	// hazard Jackson's primitive default was in the published Java SDK 1.0.8:
	// json.Unmarshal does NOT error on a missing key, so `Blocked bool` stays
	// false and every caller's `if resp.Blocked { deny }` reads ALLOW for a
	// 200 that is not a verdict at all — schema drift, a proxy/WAF
	// substituting its own envelope on a 2xx, a truncated body, `{}`,
	// `{"data":null}`. There are THREE outcomes, not two: blocked, allowed,
	// and NO VERDICT. A response the client cannot INTERPRET must DENY.
	ErrCodeIndeterminate ErrorCode = "INDETERMINATE_VERDICT"
)

// indeterminateVerdict builds the refusal returned when a 2xx response carries
// no decision field. The message names the MISSING field so an operator can
// tell schema drift from an outage at a glance, and callers can branch on
// ErrCodeIndeterminate.
func indeterminateVerdict(method, field, route string) error {
	return &EvalGuardError{
		Code: ErrCodeIndeterminate,
		Message: fmt.Sprintf(
			"%s: %s returned 2xx with no `%s` field, so the security verdict is INDETERMINATE — "+
				"the content was NOT evaluated and must not be treated as allowed",
			method, route, field),
	}
}

// uninterpretableVerdict is the SAME refusal for the other half of the class:
// the decision field is PRESENT but the client cannot act on it — a verdict
// string outside the server's closed set, or a verdict that contradicts the
// rest of the body. It deliberately carries ErrCodeIndeterminate too: from a
// caller's point of view "no verdict" and "a verdict I cannot interpret" are
// the same third outcome, and a caller already branching on
// ErrCodeIndeterminate must catch both without a code change.
//
// AUDIT 2026-08-03. Presence was being checked where VALIDITY was required.
// `HasVerdict()` returning true for ANY non-empty string meant a case-drifted
// or newer verdict ("Block", "PASS", "quarantine") satisfied the gate and a
// server the audit scored 98/100 got deployed. Same shape as the RAG scan
// reporting poison it could not attribute to any document.
func uninterpretableVerdict(method, route, reason string) error {
	return &EvalGuardError{
		Code: ErrCodeIndeterminate,
		Message: fmt.Sprintf(
			"%s: %s returned 2xx but the security verdict is UNINTERPRETABLE (%s) — "+
				"the result must NOT be treated as allowed",
			method, route, reason),
	}
}

// quoteVerdict renders an untrusted verdict string for an operator message.
// Bounded on purpose: the value is attacker-influenceable server output that
// lands in application logs, so it is truncated rather than echoed whole.
func quoteVerdict(s string) string {
	const max = 48
	if len(s) > max {
		return fmt.Sprintf("%q(+%d more bytes)", s[:max], len(s)-max)
	}
	return fmt.Sprintf("%q", s)
}

// EvalGuardError is the base error type for all SDK errors.
type EvalGuardError struct {
	Code       ErrorCode `json:"code"`
	Message    string    `json:"message"`
	StatusCode int       `json:"status_code,omitempty"`
	RequestID  string    `json:"request_id,omitempty"`
}

func (e *EvalGuardError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf("evalguard: %s (code=%s, status=%d, request_id=%s)", e.Message, e.Code, e.StatusCode, e.RequestID)
	}
	return fmt.Sprintf("evalguard: %s (code=%s, status=%d)", e.Message, e.Code, e.StatusCode)
}

// AuthError indicates authentication or authorization failure.
type AuthError struct {
	EvalGuardError
}

// RateLimitError indicates the client has been rate-limited.
type RateLimitError struct {
	EvalGuardError
	RetryAfter time.Duration
}

// Option configures the Client.
type Option func(*Client)

// WithBaseURL sets a custom API base URL. The value is validated and normalized
// when the client is built (see NewClient): a plaintext http:// base for a
// non-loopback host is rejected because every request sends
// "Authorization: Bearer <apiKey>" and a cleartext URL would leak the key.
// Trailing slashes are stripped, and the "/v1" version segment is appended when
// missing — so "https://evalguard.ai/api" normalizes to
// "https://evalguard.ai/api/v1" instead of 404ing every call (every request
// path here is version-less, e.g. "/evals"). Mirrors the TS/CLI/Python SDKs.
func WithBaseURL(url string) Option {
	return func(c *Client) { c.baseURL = url }
}

// WithTimeout sets the HTTP client timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) { c.httpClient.Timeout = d }
}

// WithHTTPClient sets a custom http.Client.
func WithHTTPClient(hc *http.Client) Option {
	return func(c *Client) { c.httpClient = hc }
}

// Client is the EvalGuard API client.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client

	// resolvedProjectID caches the default project resolved from
	// GET /project/current the first time a project-scoped method is called
	// without an explicit projectId. projectMu guards the cache so repeated
	// calls across goroutines resolve at most once.
	projectMu         sync.Mutex
	resolvedProjectID string
}

// secureBaseURL validates and normalizes an API base URL. It mirrors
// assertSecureBaseUrl in wrapper-core (guardrail-client.ts) and the Python
// client (client.py): a plaintext http:// base for a NON-loopback host is
// refused because every request carries "Authorization: Bearer <apiKey>", so a
// cleartext URL would leak the key and let a MITM tamper with responses.
// http://localhost / 127.0.0.1 / ::1 stay allowed for local development;
// https:// and wss:// always pass. Trailing slashes are stripped so that
// baseURL+path never produces a double slash.
//
// Version normalization (JS-B6 parity): every request path in this client is
// version-less (e.g. "/evals", "/ai-sbom/generate"), so the base must terminate
// at the "/v1" version segment. When the configured base ends in "/api" — one
// segment short of the versioned root — "/v1" is appended, so a base of
// "https://evalguard.ai/api" resolves to "https://evalguard.ai/api/v1" instead
// of 404ing every call. A base that already ends in "/v1" (the default) is left
// as-is, and any other path is left untouched (an explicit "/api/v2", a bare
// host used by local mocks, etc.) so normalization never rewrites a URL the
// caller deliberately pointed at.
func secureBaseURL(base string) (string, error) {
	trimmed := strings.TrimRight(base, "/")
	u, err := url.Parse(trimmed)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return "", &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("baseURL is not a valid URL: %q", base)}
	}
	// url.Hostname strips the brackets from an IPv6 literal (e.g. "[::1]" -> "::1").
	host := u.Hostname()
	isLoopback := host == "localhost" || host == "127.0.0.1" || host == "::1"
	if u.Scheme == "http" && !isLoopback {
		return "", &EvalGuardError{
			Code:    ErrCodeValidation,
			Message: fmt.Sprintf("baseURL must use https:// (got %q) — a plaintext URL would leak your API key; use https, or http://localhost for local testing", base),
		}
	}
	// Compare on the URL path (not the whole string) so a host that merely ends
	// in the characters "api"/"v1" is never mistaken for a path segment.
	path := u.EscapedPath()
	if !strings.HasSuffix(path, "/v1") && strings.HasSuffix(path, "/api") {
		trimmed += "/v1"
	}
	return trimmed, nil
}

// NewClient creates a new EvalGuard client.
func NewClient(apiKey string, opts ...Option) (*Client, error) {
	if apiKey == "" {
		return nil, &EvalGuardError{Code: ErrCodeUnauthorized, Message: "api key is required"}
	}
	c := &Client{
		apiKey:  apiKey,
		baseURL: DefaultBaseURL,
		httpClient: &http.Client{
			Timeout: DefaultTimeout,
		},
	}
	for _, o := range opts {
		o(c)
	}
	// Reject an insecure (plaintext, non-loopback) base URL now that any
	// WithBaseURL override has been applied — WithBaseURL can't return an error
	// itself, so the option's contract is enforced here at build time.
	normalized, err := secureBaseURL(c.baseURL)
	if err != nil {
		return nil, err
	}
	c.baseURL = normalized
	return c, nil
}

// --- Request / Response types ---

// EvalCase is a single test case in an evaluation run.
type EvalCase struct {
	Input          string `json:"input"`
	ExpectedOutput string `json:"expectedOutput,omitempty"`
}

// RunEvalRequest contains parameters for starting an evaluation run.
// Mirrors createEvalSchema on POST /api/v1/evals (requiredRole editor):
// at least one case and one scorer are required.
type RunEvalRequest struct {
	Name      string     `json:"name"`
	ProjectID string     `json:"projectId"`
	Model     string     `json:"model"`
	Prompt    string     `json:"prompt"`
	Cases     []EvalCase `json:"cases"`
	Scorers   []string   `json:"scorers"`
}

// EvalRunStarted is the 201 payload returned by RunEval. The run executes
// asynchronously in the background; poll GetEval(ID) for results.
type EvalRunStarted struct {
	ID         string `json:"id"`
	Status     string `json:"status"`
	TotalTests int    `json:"totalTests"`
	Model      string `json:"model"`
	Message    string `json:"message"`
}

// EvalResult is a single eval-run row as returned by GetEval and ListEvals.
// Score and CompletedAt are nil while the run is still in progress; Config
// is the opaque run configuration ({prompt, cases, scorers}) preserved as
// raw JSON. Duration is wall-clock milliseconds, set once the run finishes.
type EvalResult struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Model       string          `json:"model"`
	Status      string          `json:"status"`
	Score       *float64        `json:"score"`
	Config      json.RawMessage `json:"config"`
	CreatedAt   time.Time       `json:"created_at"`
	CompletedAt *time.Time      `json:"completed_at,omitempty"`
	Duration    *int            `json:"duration,omitempty"`
	Error       *string         `json:"error,omitempty"`
}

// EvalScore is a single scorer's verdict for one executed test case.
type EvalScore struct {
	Score  float64 `json:"score"`
	Passed bool    `json:"passed"`
	Reason string  `json:"reason,omitempty"`
}

// EvalCaseResult is one executed test case in a finished eval run.
type EvalCaseResult struct {
	ID            string               `json:"id"`
	TestCaseIndex int                  `json:"test_case_index"`
	Input         string               `json:"input"`
	Expected      *string              `json:"expected,omitempty"`
	Output        string               `json:"output"`
	Scores        map[string]EvalScore `json:"scores"`
	Score         float64              `json:"score"`
	LatencyMs     float64              `json:"latency_ms"`
	Cost          float64              `json:"cost"`
	Passed        bool                 `json:"passed"`
}

// EvalSummary aggregates a run's per-case results.
type EvalSummary struct {
	TotalCases   int     `json:"totalCases"`
	PassedCases  int     `json:"passedCases"`
	FailedCases  int     `json:"failedCases"`
	PassRate     float64 `json:"passRate"`
	AvgScore     float64 `json:"avgScore"`
	TotalLatency float64 `json:"totalLatency"`
	TotalCost    float64 `json:"totalCost"`
}

// EvalRunDetail is the full GetEval payload — the run row plus its executed
// cases and aggregate summary, as returned by GET /api/v1/evals/{id}.
type EvalRunDetail struct {
	Run     EvalResult       `json:"run"`
	Results []EvalCaseResult `json:"results"`
	Summary EvalSummary      `json:"summary"`
}

// SecurityScanRequest contains parameters for a security scan.
//
// The server's POST /api/v1/security route validates the body with
// createSecurityScanSchema, which REQUIRES projectId (UUID), model,
// prompt (a single non-empty string), and attackTypes (1..50 entries).
// The previous DTO sent {prompts, scan_types, severity} which the schema
// rejected with 400 VALIDATION_ERROR on every call — the scan never ran.
type SecurityScanRequest struct {
	// ProjectID is the UUID of the project the scan belongs to (required).
	ProjectID string `json:"projectId"`
	// Model is the target model identifier, e.g. "gpt-4o" (required).
	Model string `json:"model"`
	// Prompt is the single prompt/system instruction to attack (required).
	Prompt string `json:"prompt"`
	// AttackTypes is the list of attack categories to exercise, e.g.
	// ["prompt-injection", "jailbreak"] (1..50 entries, required).
	AttackTypes []string `json:"attackTypes"`
	// Depth selects how many evasion STRATEGIES the scan runs: "quick" (4),
	// "standard" (16), or "full" (every shipped strategy). Optional.
	//
	// AUDIT 2026-08-06 — this field did not exist, and its absence was a
	// live fail-open. The server's DEFAULT_SCAN_DEPTH is "full", and only a set
	// within SYNC_SCAN_STRATEGY_BUDGET (4, the "quick" set) may run inside the
	// request; anything larger is QUEUED and answered 202 `status:"pending"`
	// with no score and no counts. So every Go call resolved to "full", went
	// async, and decoded to Score 0 / TotalTests 0 / SeverityCounts all-zero
	// with err == nil — a build gate reading `SeverityCounts.Critical > 0`
	// passed every time. Set Depth "quick" for an inline verdict; leave it empty
	// (or "standard"/"full") and poll the queued scan by ID.
	Depth string `json:"depth,omitempty"`
	// StrategyIDs runs an explicit strategy set instead of a named Depth.
	// Optional; when non-empty it wins over Depth.
	StrategyIDs []string `json:"strategyIds,omitempty"`
}

// SecuritySeverityCounts is the per-severity finding tally returned by a scan.
type SecuritySeverityCounts struct {
	Critical int `json:"critical"`
	High     int `json:"high"`
	Medium   int `json:"medium"`
	Low      int `json:"low"`
}

// SecurityScanResult is the output of a security scan.
//
// POST /api/v1/security answers in ONE OF TWO SHAPES, which is the fact this
// struct hid until 2026-08-06:
//
//   - 201, Mode "sync": the scan ran inline. Status is "passed" or "failed",
//     and Score / TotalTests / SeverityCounts / FindingsCount are the verdict.
//     Reached only when the resolved strategy set fits the sync budget — i.e.
//     Depth "quick".
//   - 202, Mode "async": the scan was QUEUED. Status is "pending", StatusURL
//     says where to poll, and every count field is ABSENT — so they decode to
//     zero. This is the shape every Go call got, because the server's default
//     depth is "full".
//
// Queued() and HasVerdict() separate the two. RunSecurityScan refuses the
// queued shape outright rather than hand back an all-zero summary that reads
// like a clean scan. Individual findings are never inlined — fetch them from
// the scan detail endpoint by ID.
type SecurityScanResult struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	// Mode is "sync" (ran inline) or "async" (queued).
	Mode string `json:"mode,omitempty"`
	// StatusURL is where to poll a queued scan. Set only on the async shape.
	StatusURL      string                 `json:"statusUrl,omitempty"`
	Score          float64                `json:"score"`
	TotalTests     int                    `json:"totalTests"`
	ExecutedTests  int                    `json:"executedTests"`
	ErroredTests   int                    `json:"erroredTests"`
	Duration       float64                `json:"duration"`
	SeverityCounts SecuritySeverityCounts `json:"severityCounts"`
	FindingsCount  int                    `json:"findingsCount"`
}

// Trace represents a single observability trace SUMMARY as returned by
// GET /api/v1/traces. The fields mirror the server's aggregated row exactly
// (dbRowToSummary in apps/web/src/app/api/v1/traces/route.ts):
// { traceId, rootSpanName, duration, spanCount, services, status, startTime }.
//
// The previous struct used snake_case fields (id/parent_id/name/start_time/
// duration_ms/tokens_in/…) that the API never emits, so EVERY trace decoded to
// an all-zero struct. StartTime/Duration are epoch/ms numbers (not RFC3339),
// matching the server, so they're plain int64s here.
type Trace struct {
	// TraceID is the trace's id (groups all spans).
	TraceID string `json:"traceId"`
	// RootSpanName is the name of the root span (the entry point).
	RootSpanName string `json:"rootSpanName"`
	// Duration is the wall-clock span of the trace in milliseconds
	// (max end_time_ms − min start_time_ms across its spans).
	Duration int64 `json:"duration"`
	// SpanCount is the number of spans in the trace.
	SpanCount int `json:"spanCount"`
	// Services is the distinct set of service.name values seen in the trace.
	Services []string `json:"services"`
	// Status is "ok" | "error" | "unset" — "error" if any span errored.
	Status string `json:"status"`
	// StartTime is the trace's earliest span start, epoch milliseconds.
	StartTime int64 `json:"startTime"`
}

// GetTracesRequest contains filters for listing traces. These are sent as
// QUERY PARAMETERS (GET /api/v1/traces?projectId=...&...), not a JSON body —
// the API reads request.nextUrl.searchParams. ProjectID is REQUIRED (the API
// returns 400 "projectId is required" without it).
type GetTracesRequest struct {
	ProjectID   string    // required
	StartTime   time.Time // matched against start_time_ms (sent as epoch millis)
	EndTime     time.Time
	Model       string
	ServiceName string
	Status      string // "ok" | "error" | "unset"
	Cursor      string // opaque keyset cursor from a prior NextCursor
	Limit       int
	Offset      int // legacy offset pagination (prefer Cursor)
}

// GetTracesResponse is a paginated list of traces. Field names match the API
// envelope's data object: { traces, total, nextCursor }.
type GetTracesResponse struct {
	Traces     []Trace `json:"traces"`
	Total      int     `json:"total"`
	NextCursor string  `json:"nextCursor"`
}

// HasMore reports whether another page is available (a non-empty keyset cursor).
func (r *GetTracesResponse) HasMore() bool { return r.NextCursor != "" }

// CreateDatasetRequest contains parameters for creating a dataset. Matches the
// POST /api/v1/datasets body: ProjectID is REQUIRED (a UUID), rows live under
// "cases" (not "items"), and "source" is an optional origin tag (default
// "manual" server-side). The previous struct sent {name,items,tags} with no
// projectId and 400'd on every call.
type CreateDatasetRequest struct {
	Name        string        `json:"name"`
	ProjectID   string        `json:"projectId"`
	Description string        `json:"description,omitempty"`
	Source      string        `json:"source,omitempty"`
	Cases       []DatasetItem `json:"cases,omitempty"`
}

// DatasetItem represents a single row in a dataset.
type DatasetItem struct {
	Input          string         `json:"input"`
	ExpectedOutput string         `json:"expectedOutput,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

// Dataset represents a stored dataset.
type Dataset struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	ItemCount   int               `json:"item_count"`
	Tags        map[string]string `json:"tags,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// --- Project auto-resolution ---

// projectCurrent is the raw payload of GET /api/v1/project/current. The
// endpoint returns this object UNWRAPPED (not under the {success,data}
// envelope); unmarshalEnvelope's fallback decodes the whole body for us.
type projectCurrent struct {
	ProjectID string `json:"projectId"`
	OrgID     string `json:"orgId"`
}

// resolveProjectID returns the caller's default project id, fetching it from
// GET /api/v1/project/current the first time and caching it on the client so
// subsequent calls don't re-fetch. On a fresh org the endpoint auto-creates a
// default project. It errors if no project could be resolved so callers see an
// actionable message instead of a downstream 400.
func (c *Client) resolveProjectID(ctx context.Context) (string, error) {
	c.projectMu.Lock()
	defer c.projectMu.Unlock()
	if c.resolvedProjectID != "" {
		return c.resolvedProjectID, nil
	}

	var pc projectCurrent
	if err := c.doRequest(ctx, http.MethodGet, "/project/current", nil, &pc); err != nil {
		return "", fmt.Errorf("resolveProjectID: %w", err)
	}
	if pc.ProjectID == "" {
		return "", &EvalGuardError{
			Code:    ErrCodeValidation,
			Message: "could not resolve a default project; pass projectId explicitly",
		}
	}

	c.resolvedProjectID = pc.ProjectID
	return c.resolvedProjectID, nil
}

// --- API methods ---

// RunEval starts an asynchronous evaluation run.
//
// POST /api/v1/evals creates the run and fires background execution,
// returning 201 immediately with the new run's ID and status "running".
// The run does NOT finish before this call returns — poll GetEval(ID)
// for status, score, and results.
func (c *Client) RunEval(ctx context.Context, req *RunEvalRequest) (*EvalRunStarted, error) {
	if req == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunEval: req is required"}
	}
	// When the caller omits projectId, resolve (and cache) the default project.
	// An explicitly-set ProjectID always wins and skips the lookup.
	if req.ProjectID == "" {
		pid, err := c.resolveProjectID(ctx)
		if err != nil {
			return nil, fmt.Errorf("RunEval: %w", err)
		}
		req.ProjectID = pid
	}
	var result EvalRunStarted
	if err := c.doRequest(ctx, http.MethodPost, "/evals", req, &result); err != nil {
		return nil, fmt.Errorf("RunEval: %w", err)
	}
	return &result, nil
}

// GetEval retrieves a single eval run by ID. GET /api/v1/evals/{id}.
func (c *Client) GetEval(ctx context.Context, evalID string) (*EvalRunDetail, error) {
	var result EvalRunDetail
	if err := c.doRequest(ctx, http.MethodGet, "/evals/"+evalID, nil, &result); err != nil {
		return nil, fmt.Errorf("GetEval: %w", err)
	}
	return &result, nil
}

// ListEvals returns the eval runs for a project, newest first.
// GET /api/v1/evals?projectId=... — projectId is required and the
// response payload is a bare array of run rows.
func (c *Client) ListEvals(ctx context.Context, projectID string) ([]EvalResult, error) {
	// When the caller omits projectID, resolve (and cache) the default project.
	// An explicitly-passed projectID always wins and skips the lookup.
	if projectID == "" {
		pid, err := c.resolveProjectID(ctx)
		if err != nil {
			return nil, fmt.Errorf("ListEvals: %w", err)
		}
		projectID = pid
	}
	var result []EvalResult
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/evals?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListEvals: %w", err)
	}
	return result, nil
}

// RunSecurityScan runs a security scan synchronously and returns its summary.
//
// The server validates projectId/model/prompt/attackTypes before running;
// we validate the same invariants client-side so callers get an actionable
// error instead of an opaque 400 from the wire.
func (c *Client) RunSecurityScan(ctx context.Context, req *SecurityScanRequest) (*SecurityScanResult, error) {
	if req == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunSecurityScan: req is required"}
	}
	// When the caller omits projectId, resolve (and cache) the default project.
	// An explicitly-set ProjectID always wins and skips the lookup.
	if req.ProjectID == "" {
		pid, err := c.resolveProjectID(ctx)
		if err != nil {
			return nil, fmt.Errorf("RunSecurityScan: %w", err)
		}
		req.ProjectID = pid
	}
	if req.Model == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunSecurityScan: Model is required"}
	}
	if req.Prompt == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunSecurityScan: Prompt is required"}
	}
	if len(req.AttackTypes) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunSecurityScan: at least one AttackType is required"}
	}
	var result SecurityScanResult
	// Security scans are created at POST /security (there is no /security/scan).
	if err := c.doRequest(ctx, http.MethodPost, "/security", req, &result); err != nil {
		return nil, fmt.Errorf("RunSecurityScan: %w", err)
	}
	// Every numeric field zero-values to 0 and Status to "", so a 2xx that was
	// not a scan read as "0 tests, 0 findings, 0 critical" — a clean red-team
	// for a run that never happened. Structural twin of RunAgentExecRedTeam,
	// which 1.5.0 hardened; this one was missed.
	if result.Status == "" {
		return nil, indeterminateVerdict("RunSecurityScan", "status", "POST /security")
	}
	if reason := result.inconsistency(); reason != "" {
		return nil, uninterpretableVerdict("RunSecurityScan", "POST /security", reason)
	}
	// A QUEUED scan is the third outcome: not passed, not failed, no verdict
	// yet. It is also what every Go call got before Depth existed, and its
	// all-zero counts are indistinguishable from a clean result — so it is
	// refused rather than returned. The id and poll URL travel in the message so
	// nothing operational is lost.
	if result.Queued() {
		return nil, &EvalGuardError{
			Code: ErrCodeIndeterminate,
			Message: fmt.Sprintf(
				"RunSecurityScan: the scan did not run inline — POST /security queued it as %q and "+
					"answered 202 `status:\"pending\"` with no score and no findings, so there is NO "+
					"verdict to read (an all-zero summary here is a scan that has not started, not a "+
					"clean one). Poll %q for the result, or set Depth:\"quick\" to run the scan inside "+
					"the request.",
				result.ID, result.StatusURL),
		}
	}
	return &result, nil
}

// GetTraces retrieves observability traces. Filters are sent as query
// parameters (the API reads searchParams); the previous implementation
// json-marshalled them into a GET body that the server ignored, so every
// filter was silently dropped and results came back wrong/empty.
func (c *Client) GetTraces(ctx context.Context, req *GetTracesRequest) (*GetTracesResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("GetTraces: req is required")
	}
	// projectId is required server-side (400 without it). When the caller omits it,
	// resolve (and cache) the default project, mirroring RunEval/ListEvals.
	if req.ProjectID == "" {
		pid, err := c.resolveProjectID(ctx)
		if err != nil {
			return nil, fmt.Errorf("GetTraces: %w", err)
		}
		req.ProjectID = pid
	}
	q := url.Values{}
	q.Set("projectId", req.ProjectID)
	if !req.StartTime.IsZero() {
		q.Set("startTime", strconv.FormatInt(req.StartTime.UnixMilli(), 10))
	}
	if !req.EndTime.IsZero() {
		q.Set("endTime", strconv.FormatInt(req.EndTime.UnixMilli(), 10))
	}
	if req.Model != "" {
		q.Set("model", req.Model)
	}
	if req.ServiceName != "" {
		q.Set("serviceName", req.ServiceName)
	}
	if req.Status != "" {
		q.Set("status", req.Status)
	}
	if req.Cursor != "" {
		q.Set("cursor", req.Cursor)
	}
	if req.Limit > 0 {
		q.Set("limit", strconv.Itoa(req.Limit))
	}
	if req.Offset > 0 {
		q.Set("offset", strconv.Itoa(req.Offset))
	}
	path := "/traces"
	if enc := q.Encode(); enc != "" {
		path += "?" + enc
	}

	var result GetTracesResponse
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("GetTraces: %w", err)
	}
	return &result, nil
}

// CreateDataset creates a new evaluation dataset.
func (c *Client) CreateDataset(ctx context.Context, req *CreateDatasetRequest) (*Dataset, error) {
	var result Dataset
	if err := c.doRequest(ctx, http.MethodPost, "/datasets", req, &result); err != nil {
		return nil, fmt.Errorf("CreateDataset: %w", err)
	}
	return &result, nil
}

// --- Dataset versioning (Phase 6b, 2026-05-22) ---
//
// Immutable per-dataset snapshots for reproducible evals. Same surface
// as Python/Java/Node SDKs. Returns map[string]any rather than typed
// DTOs so the contract can evolve quickly; callers cast keys they need.

// ListDatasetVersions returns immutable snapshots for a dataset, newest first.
func (c *Client) ListDatasetVersions(ctx context.Context, datasetID string) (map[string]any, error) {
	var result map[string]any
	path := fmt.Sprintf("/datasets/%s/versions", datasetID)
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("ListDatasetVersions: %w", err)
	}
	return result, nil
}

// SnapshotDataset records the dataset's current cases as a new immutable
// version. Returns {unchanged: true, version} when content hash matches
// the latest version (no new row written).
//
// description is optional; pass "" to skip.
func (c *Client) SnapshotDataset(ctx context.Context, datasetID, description string) (map[string]any, error) {
	var result map[string]any
	body := map[string]any{}
	if description != "" {
		body["description"] = description
	}
	path := fmt.Sprintf("/datasets/%s/versions", datasetID)
	if err := c.doRequest(ctx, http.MethodPost, path, body, &result); err != nil {
		return nil, fmt.Errorf("SnapshotDataset: %w", err)
	}
	return result, nil
}

// GetDatasetVersion fetches a single snapshot including its inline cases payload.
func (c *Client) GetDatasetVersion(ctx context.Context, datasetID, versionID string) (map[string]any, error) {
	var result map[string]any
	path := fmt.Sprintf("/datasets/%s/versions/%s", datasetID, versionID)
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("GetDatasetVersion: %w", err)
	}
	return result, nil
}

// RestoreDatasetVersion restores a dataset to a frozen version. The
// endpoint auto-snapshots the pre-restore state first so the operation
// is reversible. Returns {restoredFromVersion, caseCount, preRestoreVersionNum}.
func (c *Client) RestoreDatasetVersion(ctx context.Context, datasetID, versionID string) (map[string]any, error) {
	var result map[string]any
	path := fmt.Sprintf("/datasets/%s/versions/%s/restore", datasetID, versionID)
	if err := c.doRequest(ctx, http.MethodPost, path, map[string]any{}, &result); err != nil {
		return nil, fmt.Errorf("RestoreDatasetVersion: %w", err)
	}
	return result, nil
}

// DiffDatasetVersions returns added/removed/modified/unchanged counts +
// the first-10 sample changes between two snapshots of the same dataset.
func (c *Client) DiffDatasetVersions(ctx context.Context, datasetID, fromVersionID, toVersionID string) (map[string]any, error) {
	var result map[string]any
	path := fmt.Sprintf("/datasets/%s/versions/%s/diff?to=%s", datasetID, fromVersionID, toVersionID)
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("DiffDatasetVersions: %w", err)
	}
	return result, nil
}

// ── Evaluator Hub (versioned, reusable evaluator registry) ──────────
//
// Content-hash registry: one row per (project, name, version), content-hash
// deduped. Mirrors the TS/Python SDKs + the `evalguard evaluators` CLI.

// ListEvaluators returns evaluator versions for a project (newest first).
// Pass name="" for all evaluators, or a name for one evaluator's full history.
//
// GET /api/v1/evaluators replies with apiSuccess(data) where data is a BARE
// ARRAY of evaluator-version rows. The previous map[string]any target left the
// result nil/empty on every call because the envelope's "data" is a JSON array,
// not an object — json.Unmarshal of [] into a map is a no-op.
func (c *Client) ListEvaluators(ctx context.Context, projectID, name string) ([]map[string]any, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "projectID is required"}
	}
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if name != "" {
		q.Set("name", name)
	}
	if err := c.doRequest(ctx, http.MethodGet, "/evaluators?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListEvaluators: %w", err)
	}
	return result, nil
}

// CreateEvaluatorRequest is the body for CreateEvaluator. Definition is
// {"kind": "llm-judge"|"code"|"heuristic"|"composite", "config": {...}, "threshold": float}.
type CreateEvaluatorRequest struct {
	ProjectID  string         `json:"projectId"`
	Name       string         `json:"name"`
	Definition map[string]any `json:"definition"`
	Notes      string         `json:"notes,omitempty"`
	Activate   *bool          `json:"activate,omitempty"`
}

// CreateEvaluator creates a new evaluator version (content-hash deduped against the latest).
func (c *Client) CreateEvaluator(ctx context.Context, req *CreateEvaluatorRequest) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/evaluators", req, &result); err != nil {
		return nil, fmt.Errorf("CreateEvaluator: %w", err)
	}
	return result, nil
}

// DiffEvaluatorVersions returns the field-level diff between two versions of a named evaluator.
func (c *Client) DiffEvaluatorVersions(ctx context.Context, projectID, name string, fromVersion, toVersion int) (map[string]any, error) {
	var result map[string]any
	body := map[string]any{
		"projectId":   projectID,
		"name":        name,
		"fromVersion": fromVersion,
		"toVersion":   toVersion,
	}
	if err := c.doRequest(ctx, http.MethodPost, "/evaluators/diff", body, &result); err != nil {
		return nil, fmt.Errorf("DiffEvaluatorVersions: %w", err)
	}
	return result, nil
}

// ── Scorer calibration (CLHF — continuous learning from human feedback) ──

// CalibrateScorerRequest is the body for CalibrateScorer. Provide Pairs
// ([{"human": bool, "machine": bool}]) and/or Scored
// ([{"humanPass": bool, "machineScore": float}]).
type CalibrateScorerRequest struct {
	Pairs            []map[string]bool `json:"pairs,omitempty"`
	Scored           []map[string]any  `json:"scored,omitempty"`
	ProjectID        string            `json:"projectId,omitempty"`
	ScorerID         string            `json:"scorerId,omitempty"`
	CurrentThreshold *float64          `json:"currentThreshold,omitempty"`
}

// CalibrateScorer quantifies evaluator/human agreement (chance-corrected
// Cohen's kappa) and recommends the best score threshold.
func (c *Client) CalibrateScorer(ctx context.Context, req *CalibrateScorerRequest) (map[string]any, error) {
	if len(req.Pairs) == 0 && len(req.Scored) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "CalibrateScorer: provide at least one of Pairs or Scored"}
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/scorers/calibrate", req, &result); err != nil {
		return nil, fmt.Errorf("CalibrateScorer: %w", err)
	}
	return result, nil
}

// Scorer is a platform scorer (a built-in evaluator the eval engine can run).
type Scorer struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Type        string          `json:"type,omitempty"`
	Config      json.RawMessage `json:"config,omitempty"`
}

// ListScorers returns every scorer the platform exposes. Mirrors the npm/Python
// listScorers — Go (and Java) previously had no scorer-listing method at all.
//
// GET /scorers replies with an enveloped object ({data: {scorers: [...], total}}),
// so this unwraps data.scorers rather than treating data as a bare array.
func (c *Client) ListScorers(ctx context.Context) ([]Scorer, error) {
	var payload struct {
		Scorers []Scorer `json:"scorers"`
	}
	if err := c.doRequest(ctx, http.MethodGet, "/scorers", nil, &payload); err != nil {
		return nil, fmt.Errorf("ListScorers: %w", err)
	}
	return payload.Scorers, nil
}

// --- Shadow AI ---

// ShadowAIRequest contains parameters for shadow AI analysis.
type ShadowAIRequest struct {
	Input    string `json:"input"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// ShadowAIResult is the output of a shadow AI analysis.
type ShadowAIResult struct {
	Event            map[string]any `json:"event"`
	PIIDetails       map[string]any `json:"piiDetails"`
	SensitiveDetails map[string]any `json:"sensitiveDataDetails"`
}

// AnalyzeShadowAI analyzes input for shadow AI risks (PII, credentials, unauthorized models).
//
// Fails CLOSED: all three result fields are map[string]any, which decode to NIL
// for a 2xx that is not a shadow-AI analysis, so every natural read —
// res.Event["riskScore"], res.PIIDetails["detected"], res.SensitiveDetails[...] —
// returned "no PII, no credentials, no risk" for input never inspected.
func (c *Client) AnalyzeShadowAI(ctx context.Context, req *ShadowAIRequest) (*ShadowAIResult, error) {
	if req == nil || req.Input == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "AnalyzeShadowAI: req.Input is required"}
	}
	var raw json.RawMessage
	if err := c.doRequest(ctx, http.MethodPost, "/shadow-ai", req, &raw); err != nil {
		return nil, fmt.Errorf("AnalyzeShadowAI: %w", err)
	}
	// The typed shadow is decoded from the SAME bytes as the maps: the public
	// map fields cannot carry presence, and re-marshalling them would erase the
	// absent-vs-null distinction the probe exists for.
	var result ShadowAIResult
	verdict := shadowAIVerdict{inputChars: len(req.Input)}
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &result)
		_ = json.Unmarshal(raw, &verdict)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("AnalyzeShadowAI", f, "POST /shadow-ai")
	}
	if reason := verdict.inconsistency(); reason != "" {
		return nil, uninterpretableVerdict("AnalyzeShadowAI", "POST /shadow-ai", reason)
	}
	return &result, nil
}

// --- AI-SPM ---

// AIPosture represents the AI security posture.
type AIPosture struct {
	OverallScore           int            `json:"overallScore"`
	TotalModels            int            `json:"totalModels"`
	CriticalModels         int            `json:"criticalModels"`
	TotalMisconfigurations int            `json:"totalMisconfigurations"`
	DataFlows              int            `json:"dataFlows"`
	CrossBorderFlows       int            `json:"crossBorderFlows"`
	RiskDistribution       map[string]int `json:"riskDistribution"`
	Recommendations        []string       `json:"recommendations"`
}

// AIPostureResult is the full AI-SPM response.
type AIPostureResult struct {
	Posture   AIPosture        `json:"posture"`
	Models    []map[string]any `json:"models"`
	DataFlows []map[string]any `json:"dataFlows"`
}

// GetAIPosture retrieves the AI security posture management dashboard.
func (c *Client) GetAIPosture(ctx context.Context, projectID string) (*AIPostureResult, error) {
	var result AIPostureResult
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/ai-spm?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetAIPosture: %w", err)
	}
	return &result, nil
}

// --- Smart Copilot ---

// CopilotAnalyzeRequest contains parameters for copilot analysis.
//
// The server (POST /api/v1/copilot/analyze) requires specific shapes per Type
// and 400s on anything else:
//   - Type MUST be one of "security", "eval", or "contextual".
//   - "security"/"contextual" require Findings; each finding needs
//     {type, severity, title, description, passed(bool)} (pluginId optional).
//   - "eval" requires Cases; each case needs a string "input", a numeric
//     "score", and a boolean "passed" (scorerType optional). A case missing the
//     numeric score is rejected — these maps are free-form on the Go side, so
//     populate score/passed explicitly.
type CopilotAnalyzeRequest struct {
	Type     string           `json:"type"` // "security" | "eval" | "contextual"
	Model    string           `json:"model"`
	PassRate float64          `json:"passRate,omitempty"`
	Score    float64          `json:"score,omitempty"`
	Findings []map[string]any `json:"findings,omitempty"`
	// Cases entries each require a string "input", a numeric "score", and a
	// boolean "passed" (e.g. {"input": "2+2?", "score": 1, "passed": true}).
	Cases []map[string]any `json:"cases,omitempty"`
}

// CopilotAnalyzeResult is the copilot analysis output.
type CopilotAnalyzeResult struct {
	Type     string         `json:"type"`
	Analysis map[string]any `json:"analysis"`
}

// AnalyzeCopilot runs the smart copilot to analyze security or eval results.
func (c *Client) AnalyzeCopilot(ctx context.Context, req *CopilotAnalyzeRequest) (*CopilotAnalyzeResult, error) {
	var result CopilotAnalyzeResult
	if err := c.doRequest(ctx, http.MethodPost, "/copilot/analyze", req, &result); err != nil {
		return nil, fmt.Errorf("AnalyzeCopilot: %w", err)
	}
	return &result, nil
}

// --- Gateway ---

// GetGatewayHealth returns the gateway health status.
func (c *Client) GetGatewayHealth(ctx context.Context) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/gateway/health", nil, &result); err != nil {
		return nil, fmt.Errorf("GetGatewayHealth: %w", err)
	}
	return result, nil
}

// GetGatewayStats returns gateway usage statistics.
func (c *Client) GetGatewayStats(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/gateway/stats?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetGatewayStats: %w", err)
	}
	return result, nil
}

// --- Cost / FinOps ---

// GetCost returns cost analytics for a project.
func (c *Client) GetCost(ctx context.Context, projectID, period string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("period", period)
	if err := c.doRequest(ctx, http.MethodGet, "/cost?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCost: %w", err)
	}
	return result, nil
}

// GetCostForecast returns cost forecasting data.
func (c *Client) GetCostForecast(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/cost/forecast?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCostForecast: %w", err)
	}
	return result, nil
}

// --- Monitoring ---

// GetMonitoringAlerts returns active monitoring alerts.
func (c *Client) GetMonitoringAlerts(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/monitoring/alerts?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetMonitoringAlerts: %w", err)
	}
	return result, nil
}

// GetMonitoringDrift returns drift detection status.
func (c *Client) GetMonitoringDrift(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/monitoring/drift?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetMonitoringDrift: %w", err)
	}
	return result, nil
}

// --- Compliance ---

// ComplianceManualEvidence is an operator-supplied assessment of a single
// framework requirement, forwarded to POST /api/v1/compliance/check so the
// engine can fold human evidence into the automated result.
type ComplianceManualEvidence struct {
	// RequirementID is the framework requirement this evidence answers (required).
	RequirementID string `json:"requirementId"`
	// Status is the operator's determination: "met", "partial", or "not-met" (required).
	Status string `json:"status"`
	// Evidence is the free-text justification / artifact reference (required).
	Evidence string `json:"evidence"`
	// AssessedBy identifies who recorded the evidence (required).
	AssessedBy string `json:"assessedBy"`
	// AssessedAt is an optional ISO-8601 timestamp; the server stamps "now" when omitted.
	AssessedAt string `json:"assessedAt,omitempty"`
}

// ComplianceCheckRequest is the body for CheckCompliance. It mirrors the live
// POST /api/v1/compliance/check contract (PostBody in
// apps/web/src/app/api/v1/compliance/check/route.ts): the check RUNS a real
// framework assessment against a target model, so it needs the model/provider
// credentials — not just a list of framework names. The previous
// {orgId, frameworks} shape 400'd on every call.
type ComplianceCheckRequest struct {
	// OrgID is the UUID of the org the assessment belongs to (required). The
	// route resolves tenancy from this field.
	OrgID string `json:"orgId"`
	// Framework is a single framework ID to assess, e.g. "eu-ai-act" (required).
	// The endpoint validates it against its framework registry.
	Framework string `json:"framework"`
	// Model is the target model identifier to assess, e.g. "gpt-4o" (required).
	Model string `json:"model"`
	// Provider is the model provider, e.g. "openai" (required).
	Provider string `json:"provider"`
	// SystemPrompt is the system prompt the assessment probes against (required).
	SystemPrompt string `json:"systemPrompt"`
	// APIKey is the caller-supplied provider credential used to reach the model
	// during the assessment (required). It is never persisted server-side.
	APIKey string `json:"apiKey"`
	// BaseURL optionally overrides the provider endpoint (e.g. a proxy or a
	// self-hosted gateway).
	BaseURL string `json:"baseUrl,omitempty"`
	// ProjectID optionally scopes the assessment to a project (UUID).
	ProjectID string `json:"projectId,omitempty"`
	// Categories optionally restricts the assessment to specific requirement categories.
	Categories []string `json:"categories,omitempty"`
	// ManualEvidence optionally supplies operator-recorded evidence rows.
	ManualEvidence []ComplianceManualEvidence `json:"manualEvidence,omitempty"`
	// PreviousAssessmentID optionally references a prior assessment (UUID) for
	// gap/delta analysis.
	PreviousAssessmentID string `json:"previousAssessmentId,omitempty"`
	// Timeout optionally caps the assessment wall-clock in milliseconds
	// (server bounds: 1_000..600_000).
	Timeout int `json:"timeout,omitempty"`
}

// CheckCompliance runs a compliance assessment for a framework against a target
// model via POST /api/v1/compliance/check and returns the assessment result
// (including the persisted assessmentId). OrgID, Framework, Model, Provider,
// SystemPrompt, and APIKey are required by the endpoint.
func (c *Client) CheckCompliance(ctx context.Context, req *ComplianceCheckRequest) (map[string]any, error) {
	if req == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "CheckCompliance: req is required"}
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/compliance/check", req, &result); err != nil {
		return nil, fmt.Errorf("CheckCompliance: %w", err)
	}
	return result, nil
}

// --- Prompts ---

// CreatePrompt creates a new prompt version.
func (c *Client) CreatePrompt(ctx context.Context, projectID, name, content, model string) (map[string]any, error) {
	body := map[string]any{"projectId": projectID, "name": name, "content": content, "model": model}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/prompts", body, &result); err != nil {
		return nil, fmt.Errorf("CreatePrompt: %w", err)
	}
	return result, nil
}

// ListPrompts returns all prompts for a project.
func (c *Client) ListPrompts(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/prompts?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListPrompts: %w", err)
	}
	return result, nil
}

// --- Environments (Phase 2) ---
//
// Arbitrary NAMED deployment environments replace the old hardcoded
// staging/production pair (both seeded server-side for back-compat).

// ListEnvironments lists every named environment in the workspace.
func (c *Client) ListEnvironments(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/environments?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListEnvironments: %w", err)
	}
	return result, nil
}

// CreateEnvironment creates a named environment. tag defaults to "other" when
// empty; at most one environment may carry "default" (the fallback).
func (c *Client) CreateEnvironment(ctx context.Context, projectID, name, tag string) (map[string]any, error) {
	if strings.TrimSpace(name) == "" {
		return nil, fmt.Errorf("CreateEnvironment: environment name is required")
	}
	if tag == "" {
		tag = "other"
	}
	body := map[string]any{"projectId": projectID, "name": name, "tag": tag}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/environments", body, &result); err != nil {
		return nil, fmt.Errorf("CreateEnvironment: %w", err)
	}
	return result, nil
}

// RemoveEnvironment removes a named environment.
func (c *Client) RemoveEnvironment(ctx context.Context, projectID, name string) (map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodDelete, "/environments/"+url.PathEscape(name)+"?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("RemoveEnvironment: %w", err)
	}
	return result, nil
}

// SetPromptDeployment sets the deployed prompt version for the specified
// environment — the (project, environment, version) deployment mapping for the
// prompt.
func (c *Client) SetPromptDeployment(ctx context.Context, projectID, name, environment string, version int) (map[string]any, error) {
	body := map[string]any{"projectId": projectID, "environment": environment, "version": version}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/prompts/"+url.PathEscape(name)+"/deployments", body, &result); err != nil {
		return nil, fmt.Errorf("SetPromptDeployment: %w", err)
	}
	return result, nil
}

// RemovePromptDeployment removes the deployed prompt version from an environment.
func (c *Client) RemovePromptDeployment(ctx context.Context, projectID, name, environment string) (map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("environment", environment)
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodDelete, "/prompts/"+url.PathEscape(name)+"/deployments?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("RemovePromptDeployment: %w", err)
	}
	return result, nil
}

// ListPromptEnvironments lists all environments and the prompt version deployed
// to each.
func (c *Client) ListPromptEnvironments(ctx context.Context, projectID, name string) ([]map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/prompts/"+url.PathEscape(name)+"/environments?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListPromptEnvironments: %w", err)
	}
	return result, nil
}

// --- Tools (Phase 2) ---
//
// Managed, versioned Tools deployed to named environments. The config is
// validated client-side.

// CreateTool creates (or upserts a new version of) a managed Tool. config is
// validated client-side; a malformed config returns an error before any
// network call.
func (c *Client) CreateTool(ctx context.Context, projectID, name string, config ToolConfig) (map[string]any, error) {
	if ok, errs := ValidateToolConfig(config); !ok {
		return nil, fmt.Errorf("CreateTool: invalid tool config: %s", strings.Join(errs, "; "))
	}
	body := map[string]any{"projectId": projectID, "name": name, "config": config}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/tools", body, &result); err != nil {
		return nil, fmt.Errorf("CreateTool: %w", err)
	}
	return result, nil
}

// GetTool gets a Tool (latest version, or a specific version when version > 0).
func (c *Client) GetTool(ctx context.Context, projectID, name string, version int) (map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	if version > 0 {
		q.Set("version", fmt.Sprintf("%d", version))
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/tools/"+url.PathEscape(name)+"?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetTool: %w", err)
	}
	return result, nil
}

// ListTools lists all managed Tools in the workspace.
func (c *Client) ListTools(ctx context.Context, projectID string) ([]map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/tools?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListTools: %w", err)
	}
	return result, nil
}

// ListToolVersions lists every version of a Tool, ascending by version number.
func (c *Client) ListToolVersions(ctx context.Context, projectID, name string) ([]map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/tools/"+url.PathEscape(name)+"/versions?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListToolVersions: %w", err)
	}
	return result, nil
}

// SetToolDeployment sets the deployed Tool version for the specified
// environment — the (project, environment, version) deployment mapping for the
// Tool.
func (c *Client) SetToolDeployment(ctx context.Context, projectID, name, environment string, version int) (map[string]any, error) {
	body := map[string]any{"projectId": projectID, "environment": environment, "version": version}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/tools/"+url.PathEscape(name)+"/deployments", body, &result); err != nil {
		return nil, fmt.Errorf("SetToolDeployment: %w", err)
	}
	return result, nil
}

// RemoveToolDeployment removes the deployed Tool version from an environment.
func (c *Client) RemoveToolDeployment(ctx context.Context, projectID, name, environment string) (map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("environment", environment)
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodDelete, "/tools/"+url.PathEscape(name)+"/deployments?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("RemoveToolDeployment: %w", err)
	}
	return result, nil
}

// ListToolEnvironments lists all environments and the Tool version deployed to each.
func (c *Client) ListToolEnvironments(ctx context.Context, projectID, name string) ([]map[string]any, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/tools/"+url.PathEscape(name)+"/environments?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListToolEnvironments: %w", err)
	}
	return result, nil
}

// GetToolEnvironmentVariables lists a Tool's environment variables.
func (c *Client) GetToolEnvironmentVariables(ctx context.Context, projectID, name string) ([]ToolEnvironmentVariable, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []ToolEnvironmentVariable
	if err := c.doRequest(ctx, http.MethodGet, "/tools/"+url.PathEscape(name)+"/environment-variables?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetToolEnvironmentVariables: %w", err)
	}
	return result, nil
}

// AddToolEnvironmentVariable adds (or overwrites) an environment variable on a Tool.
func (c *Client) AddToolEnvironmentVariable(ctx context.Context, projectID, name, varName, value string) ([]ToolEnvironmentVariable, error) {
	if strings.TrimSpace(varName) == "" {
		return nil, fmt.Errorf("AddToolEnvironmentVariable: environment variable name is required")
	}
	body := map[string]any{"projectId": projectID, "variable": ToolEnvironmentVariable{Name: varName, Value: value}}
	var result []ToolEnvironmentVariable
	if err := c.doRequest(ctx, http.MethodPost, "/tools/"+url.PathEscape(name)+"/environment-variables", body, &result); err != nil {
		return nil, fmt.Errorf("AddToolEnvironmentVariable: %w", err)
	}
	return result, nil
}

// DeleteToolEnvironmentVariable deletes an environment variable from a Tool by name.
func (c *Client) DeleteToolEnvironmentVariable(ctx context.Context, projectID, name, varName string) ([]ToolEnvironmentVariable, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []ToolEnvironmentVariable
	if err := c.doRequest(ctx, http.MethodDelete, "/tools/"+url.PathEscape(name)+"/environment-variables/"+url.PathEscape(varName)+"?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("DeleteToolEnvironmentVariable: %w", err)
	}
	return result, nil
}

// --- Firewall ---

// FirewallCheckRequest is the body for POST /firewall/check.
//
// Input is the prompt or content to scan. Rules optionally narrows the
// detection categories (e.g. ["prompt-injection", "jailbreak"]); when nil
// or empty, all built-in layers run. ProjectID, when set, lets the server
// apply tenant-scoped overrides (custom patterns, allowlists). Subject /
// SubjectEmail are used by the consent gate when the call is made on
// behalf of an end user; both are optional.
type FirewallCheckRequest struct {
	Input        string   `json:"input"`
	Rules        []string `json:"rules,omitempty"`
	ProjectID    string   `json:"projectId,omitempty"`
	Subject      string   `json:"subject,omitempty"`
	SubjectEmail string   `json:"subject_email,omitempty"`
	SubjectID    string   `json:"subject_id,omitempty"`
}

// FirewallLayerHit reports a single detection-layer trigger from a scan.
// Layer is one of "pattern", "token", "semantic", "output", "multi-turn".
type FirewallLayerHit struct {
	Layer     string  `json:"layer"`
	Details   string  `json:"details,omitempty"`
	Score     float64 `json:"score"`
	LatencyMs float64 `json:"latencyMs"`
}

// FirewallCheckResponse is the typed response from POST /firewall/check.
//
// Blocked is true when the firewall has decided the input must not pass
// through (either via ensemble threshold or one of the forceBlockCategories).
// Score is the normalized 0..1 confidence. Category/Subcategory are the
// classifier verdicts; both are empty strings when no layer triggered.
// Hits is the per-layer breakdown of any triggered layers — an empty
// slice means the input scored below all thresholds.
//
// AUDIT 2026-08-03 (CLASS 1 fail-open, sibling of published Java 1.0.8):
// `Blocked` is a plain bool, and Go's zero value is indistinguishable from an
// explicit `blocked:false`. json.Unmarshal does not error on a missing key, so
// a 200 that is not a firewall verdict decoded cleanly into Blocked==false and
// every caller's `if resp.Blocked` read ALLOW for content the firewall never
// evaluated. Measured before the fix, a body of
// `{"success":true,"data":{"score":0.97,"category":"prompt-injection"}}`
// produced Score=0.97, Category="prompt-injection", Blocked=false, err=nil.
//
// The presence of the wire field is now tracked separately (see
// UnmarshalJSON / HasVerdict), and the client methods REFUSE rather than hand
// back a verdict they did not receive.
type FirewallCheckResponse struct {
	Blocked     bool               `json:"blocked"`
	Score       float64            `json:"score"`
	Category    string             `json:"category,omitempty"`
	Subcategory string             `json:"subcategory,omitempty"`
	LatencyMs   float64            `json:"latencyMs"`
	Hits        []FirewallLayerHit `json:"hits,omitempty"`

	// blockedPresent records whether the decoded body actually carried a
	// boolean `blocked`. Unexported so it can never be set by a caller and
	// never round-trips onto the wire.
	blockedPresent bool
}

// UnmarshalJSON decodes the response and, separately, records whether the wire
// carried a boolean `blocked` at all. An explicit `blocked:false` (a real
// allow) and an absent `blocked` (no verdict) are INDISTINGUISHABLE by value;
// only presence separates them.
//
// The `alias` indirection is what keeps this from recursing: a defined type
// with the same underlying struct has an empty method set, so the nested
// json.Unmarshal uses the default struct decoder and every other field is
// still populated exactly as before.
func (r *FirewallCheckResponse) UnmarshalJSON(data []byte) error {
	type alias FirewallCheckResponse
	var probe struct {
		Blocked *bool `json:"blocked"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = FirewallCheckResponse(decoded)
	r.blockedPresent = probe.Blocked != nil
	return nil
}

// HasVerdict reports whether the response actually carried a firewall verdict.
//
// POST /api/v1/firewall/check always emits a boolean `blocked`, so false here
// means the body did not come from the firewall. Never branch on Blocked
// without it — the client methods already refuse on your behalf, but a caller
// that decodes a stored/proxied body itself must check this.
func (r *FirewallCheckResponse) HasVerdict() bool { return r != nil && r.blockedPresent }

// CheckFirewall runs a single input through the firewall engine.
//
// This is the customer-facing one-shot check — every gateway proxy call
// exercises the same engine inline, but customers also need to call it
// directly from CI / pre-deploy hooks / rule-authoring tools.
//
// Closes finding H17 (Go SDK) from the 2026-05-07 audit: the Go SDK
// previously only had ListFirewallRules and could not actually invoke
// the firewall — Go customers had no way to call the marketed runtime
// detection. This method fills that gap with typed request/response
// structs (no map[string]any).
func (c *Client) CheckFirewall(ctx context.Context, req *FirewallCheckRequest) (*FirewallCheckResponse, error) {
	if req == nil || req.Input == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "CheckFirewall: req.Input is required"}
	}
	var result FirewallCheckResponse
	if err := c.doRequest(ctx, http.MethodPost, "/firewall/check", req, &result); err != nil {
		return nil, fmt.Errorf("CheckFirewall: %w", err)
	}
	// A 2xx that carries no `blocked` field is NOT an allow — the firewall did
	// not evaluate this input. Returning (nil, err) rather than a decoded
	// struct is deliberate: an indeterminate verdict must not be readable, or
	// the zero value becomes the answer again at the next call site.
	if !result.HasVerdict() {
		return nil, indeterminateVerdict("CheckFirewall", "blocked", "POST /firewall/check")
	}
	return &result, nil
}

// ChatCompletionsRequest is the body for POST /chat/completions (OpenAI-compatible).
type ChatCompletionsRequest struct {
	Model               string           `json:"model"`
	Messages            []map[string]any `json:"messages"`
	Temperature         *float64         `json:"temperature,omitempty"`
	TopP                *float64         `json:"top_p,omitempty"`
	MaxTokens           *int             `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int             `json:"max_completion_tokens,omitempty"`
	Stop                any              `json:"stop,omitempty"`
	// Extra carries any additional OpenAI-compatible params (forwarded verbatim).
	Extra map[string]any `json:"-"`
}

// ChatCompletions runs an OpenAI-compatible chat completion (POST /chat/completions),
// resolving the caller's BYOK provider key server-side and returning the OpenAI-exact
// response body ({id, object, created, model, choices, usage}). This is the direct
// single-provider path and works out of the box (unlike a router-config-gated gateway
// call). Streaming is NOT supported here — point the OpenAI SDK at
// {baseURL}/chat/completions for SSE streams. Parity with the TS chatCompletions().
func (c *Client) ChatCompletions(ctx context.Context, req *ChatCompletionsRequest) (map[string]any, error) {
	if req == nil || req.Model == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ChatCompletions: req.Model is required"}
	}
	if len(req.Messages) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ChatCompletions: at least one message is required"}
	}
	body := map[string]any{"model": req.Model, "messages": req.Messages, "stream": false}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.MaxTokens != nil {
		body["max_tokens"] = *req.MaxTokens
	}
	if req.MaxCompletionTokens != nil {
		body["max_completion_tokens"] = *req.MaxCompletionTokens
	}
	if req.Stop != nil {
		body["stop"] = req.Stop
	}
	for k, v := range req.Extra {
		if k != "stream" && v != nil {
			body[k] = v
		}
	}
	// The route returns the RAW OpenAI body (no {success, data} envelope).
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/chat/completions", body, &result); err != nil {
		return nil, fmt.Errorf("ChatCompletions: %w", err)
	}
	return result, nil
}

// RunGuardrails runs text through the org's guardrail policy (POST /guardrails) and
// returns {action, reasons, latencyMs} where action is allow/redact/block. Optionally
// scoped to a project's custom rules via projectID. Parity with the TS runGuardrails().
func (c *Client) RunGuardrails(ctx context.Context, text, projectID string) (map[string]any, error) {
	if text == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunGuardrails: text is required"}
	}
	body := map[string]any{"text": text}
	if projectID != "" {
		body["projectId"] = projectID
	}
	var verdict GuardrailsResult
	result, err := c.postGuardedMap(ctx, "/guardrails", body, &verdict)
	if err != nil {
		return nil, fmt.Errorf("RunGuardrails: %w", err)
	}
	// An absent `action` is "", which matches neither "block" nor "flag", so the
	// caller's gate FORWARDED the text. This is the org-policy twin of
	// CheckFirewall on the same request path and it refuses the same way.
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("RunGuardrails", f, "POST /guardrails")
	}
	if reason := verdict.inconsistency(); reason != "" {
		return nil, uninterpretableVerdict("RunGuardrails", "POST /guardrails", reason)
	}
	return result, nil
}

// SecretScanFile is one file in a multi-file ScanSecrets request.
type SecretScanFile struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// SecretScanRequest is the body for POST /security/secret-scan. Provide Content for a
// single blob (optionally with Path to locate findings) or Files for a multi-file /
// PR-diff scan. MinSeverity, when set, drops findings below that level.
type SecretScanRequest struct {
	Content     string           `json:"content,omitempty"`
	Path        string           `json:"path,omitempty"`
	Files       []SecretScanFile `json:"files,omitempty"`
	MinSeverity string           `json:"minSeverity,omitempty"`
}

// ScanSecrets scans content (or a set of files) for leaked secrets — cloud keys,
// tokens, private-key blocks, high-entropy strings — returning {scannedFiles,
// filesWithFindings, findingsCount, findings, severityCounts}. Well-known
// documentation example keys are allowlisted server-side to avoid false positives.
// Parity with the TS scanSecrets().
func (c *Client) ScanSecrets(ctx context.Context, req *SecretScanRequest) (map[string]any, error) {
	if req == nil || (req.Content == "" && len(req.Files) == 0) {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ScanSecrets: provide Content or a non-empty Files array"}
	}
	var verdict SecretScanResult
	result, err := c.postGuardedMap(ctx, "/security/secret-scan", req, &verdict)
	if err != nil {
		return nil, fmt.Errorf("ScanSecrets: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("ScanSecrets", f, "POST /security/secret-scan")
	}
	// Bound to the REQUEST: a scan cannot have opened more files than this
	// caller submitted (a single Content blob is scanned as one file).
	submitted := len(req.Files)
	if submitted == 0 {
		submitted = 1
	}
	if reason := verdict.inconsistency(submitted); reason != "" {
		return nil, uninterpretableVerdict("ScanSecrets", "POST /security/secret-scan", reason)
	}
	return result, nil
}

// ClassifyIntent classifies a prompt's intent, data-sensitivity, and governance
// risk (POST /governance/intent/classify) — a stateless, deterministic core
// classifier (no model round-trip). orgId is REQUIRED by the route (the TS SDK
// auto-resolves it from the key; the Go thin client takes it explicitly).
// sensitivityFloor is optional (public/internal/confidential/restricted).
// Parity with the TS classifyIntent().
func (c *Client) ClassifyIntent(ctx context.Context, prompt, orgID, sensitivityFloor string) (map[string]any, error) {
	if prompt == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ClassifyIntent: prompt is required"}
	}
	if orgID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ClassifyIntent: orgID is required"}
	}
	body := map[string]any{"prompt": prompt, "orgId": orgID}
	if sensitivityFloor != "" {
		body["sensitivityFloor"] = sensitivityFloor
	}
	var verdict IntentClassification
	result, err := c.postGuardedMap(ctx, "/governance/intent/classify", body, &verdict)
	if err != nil {
		return nil, fmt.Errorf("ClassifyIntent: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("ClassifyIntent", f, "POST /governance/intent/classify")
	}
	// Bound to the REQUEST: the classifier seeds sensitivity at the floor this
	// caller asked for and only ever raises it, so a lower answer is a downgrade
	// the server cannot legitimately have produced.
	if reason := verdict.inconsistency(sensitivityFloor); reason != "" {
		return nil, uninterpretableVerdict("ClassifyIntent", "POST /governance/intent/classify", reason)
	}
	return result, nil
}

// LookupVulnerabilities looks up known CVEs for a set of package URLs (purls)
// via POST /supply-chain/lookup. Parity with the TS lookupVulnerabilities().
func (c *Client) LookupVulnerabilities(ctx context.Context, purls []string) (map[string]any, error) {
	if len(purls) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "LookupVulnerabilities: purls must be a non-empty array"}
	}
	body := map[string]any{"purls": purls}
	var verdict SupplyChainLookupResult
	result, err := c.postGuardedMap(ctx, "/supply-chain/lookup", body, &verdict)
	if err != nil {
		return nil, fmt.Errorf("LookupVulnerabilities: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("LookupVulnerabilities", f, "POST /supply-chain/lookup")
	}
	// Bound to the REQUEST, entry by entry: the lookup is 1:1 with the submitted
	// purls IN ORDER, so a response about other packages cannot pass as a
	// verdict on this dependency set.
	if reason := verdict.inconsistency(purls); reason != "" {
		return nil, uninterpretableVerdict("LookupVulnerabilities", "POST /supply-chain/lookup", reason)
	}
	return result, nil
}

// IaCFile is one file in a ScanIaC request.
type IaCFile struct {
	Filename string `json:"filename"`
	Content  string `json:"content"`
}

// ScanIaC scans infrastructure-as-code files (Terraform, CloudFormation, k8s
// manifests, ...) for misconfigurations (POST /security/iac-scan). Returns
// {scannedFiles, findingsCount, bySeverity, findings}. Parity with the TS scanIac().
func (c *Client) ScanIaC(ctx context.Context, files []IaCFile) (map[string]any, error) {
	if len(files) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ScanIaC: at least one file is required"}
	}
	body := map[string]any{"files": files}
	var verdict IaCScanResult
	result, err := c.postGuardedMap(ctx, "/security/iac-scan", body, &verdict)
	if err != nil {
		return nil, fmt.Errorf("ScanIaC: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("ScanIaC", f, "POST /security/iac-scan")
	}
	// Bound to the REQUEST: scanIacFiles() sets scannedFiles to files.length
	// unconditionally, so a shortfall means the apply gate is reading a verdict
	// about only part of the infrastructure.
	if reason := verdict.inconsistency(len(files)); reason != "" {
		return nil, uninterpretableVerdict("ScanIaC", "POST /security/iac-scan", reason)
	}
	return result, nil
}

// CheckFirewallAdvanced runs the firewall with the advanced sensitivity dial and
// force-block rule categories (POST /firewall/check). sensitivity is optional
// (e.g. "permissive"/"balanced"/"strict"); rules force-block named attack
// categories. Parity with the TS checkFirewallAdvanced() (which runs the
// FirewallEngine in-process in JS; the hosted ensemble is the cross-language
// equivalent).
func (c *Client) CheckFirewallAdvanced(ctx context.Context, input string, rules []string, sensitivity string) (*FirewallCheckResponse, error) {
	if input == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "CheckFirewallAdvanced: input is required"}
	}
	body := map[string]any{"input": input}
	if len(rules) > 0 {
		body["rules"] = rules
	}
	if sensitivity != "" {
		body["sensitivity"] = sensitivity
	}
	var result FirewallCheckResponse
	if err := c.doRequest(ctx, http.MethodPost, "/firewall/check", body, &result); err != nil {
		return nil, fmt.Errorf("CheckFirewallAdvanced: %w", err)
	}
	// Same refusal as CheckFirewall — this is the second entry point onto the
	// same route, and CheckFirewallOutputAdvanced delegates here, so a gap
	// would also be a gap in model-OUTPUT screening (PII / secret leak /
	// system-prompt leak).
	if !result.HasVerdict() {
		return nil, indeterminateVerdict("CheckFirewallAdvanced", "blocked", "POST /firewall/check")
	}
	return &result, nil
}

// CheckFirewallOutputAdvanced screens MODEL OUTPUT text through the hosted firewall
// ensemble (POST /firewall/check) — PII / secret-leak / system-prompt-leak
// detection all apply to output text. (There is no dedicated hosted output route;
// the TS SDK runs FirewallEngine.scanOutput in-process, which the thin clients
// cannot replicate — this is the closest server-side equivalent.) Parity with the
// TS checkFirewallOutputAdvanced().
func (c *Client) CheckFirewallOutputAdvanced(ctx context.Context, output string, rules []string, sensitivity string) (*FirewallCheckResponse, error) {
	if output == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "CheckFirewallOutputAdvanced: output is required"}
	}
	return c.CheckFirewallAdvanced(ctx, output, rules, sensitivity)
}

// ListFirewallRules returns all firewall rules for a project.
func (c *Client) ListFirewallRules(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/firewall/rules?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListFirewallRules: %w", err)
	}
	return result, nil
}

// --- Guardrails ---

// ListGuardrails returns all guardrails for a project.
func (c *Client) ListGuardrails(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/guardrails?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListGuardrails: %w", err)
	}
	return result, nil
}

// --- Support ---

// SubmitTicket creates a support ticket.
func (c *Client) SubmitTicket(ctx context.Context, ticketType, subject, description, priority string) (map[string]any, error) {
	body := map[string]any{"type": ticketType, "subject": subject, "description": description, "priority": priority}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/support", body, &result); err != nil {
		return nil, fmt.Errorf("SubmitTicket: %w", err)
	}
	return result, nil
}

// --- Threat Intelligence ---

// GetThreatIntelligence returns the latest threat intelligence feed.
func (c *Client) GetThreatIntelligence(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/threat-intelligence?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetThreatIntelligence: %w", err)
	}
	return result, nil
}

// --- AI SBOM ---

// GetAISBOM returns the AI Software Bill of Materials.
func (c *Client) GetAISBOM(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/ai-sbom?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetAISBOM: %w", err)
	}
	return result, nil
}

// --- Team & Organization ---

// ListTeam returns team members for an organization.
func (c *Client) ListTeam(ctx context.Context, orgID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/team?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListTeam: %w", err)
	}
	return result, nil
}

// AuditLogsResponse is the data payload of GET /api/v1/audit-logs — the API
// returns apiSuccess({ logs, total }), NOT a bare array. The previous
// []map[string]any target unmarshalled the {logs,total} OBJECT into a slice
// (a no-op), so callers always got nil. Logs holds the page of rows; Total is
// the (estimated) full count for pagination.
type AuditLogsResponse struct {
	Logs  []map[string]any `json:"logs"`
	Total int              `json:"total"`
}

// GetAuditLogs returns audit logs for an organization.
func (c *Client) GetAuditLogs(ctx context.Context, orgID string) (*AuditLogsResponse, error) {
	var result AuditLogsResponse
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/audit-logs?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetAuditLogs: %w", err)
	}
	return &result, nil
}

// --- Formal Verification ---

// FormalVerifyRequest contains parameters for formal verification.
type FormalVerifyRequest struct {
	Output      string           `json:"output"`
	Constraints []map[string]any `json:"constraints"`
	Domain      string           `json:"domain,omitempty"`
}

// FormalVerify verifies AI output against formal constraints.
func (c *Client) FormalVerify(ctx context.Context, req *FormalVerifyRequest) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/formal-verification", req, &result); err != nil {
		return nil, fmt.Errorf("FormalVerify: %w", err)
	}
	return result, nil
}

// --- NL Pipeline ---

// Ask sends a natural language question to the EvalGuard NL pipeline.
func (c *Client) Ask(ctx context.Context, question, projectID string) (map[string]any, error) {
	body := map[string]any{"question": question, "projectId": projectID}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/ask", body, &result); err != nil {
		return nil, fmt.Errorf("Ask: %w", err)
	}
	return result, nil
}

// --- Leaderboard ---

// GetLeaderboard returns the public model leaderboard.
func (c *Client) GetLeaderboard(ctx context.Context, category string) (map[string]any, error) {
	path := "/leaderboard"
	if category != "" {
		q := url.Values{}
		q.Set("category", category)
		path += "?" + q.Encode()
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("GetLeaderboard: %w", err)
	}
	return result, nil
}

// --- Evals (extended) ---

// ListEvalRuns returns eval run history.
func (c *Client) ListEvalRuns(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/evals/runs?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListEvalRuns: %w", err)
	}
	return result, nil
}

// --- Security (extended) ---

// GetSecurityGraders returns available security graders.
func (c *Client) GetSecurityGraders(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/security/graders?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetSecurityGraders: %w", err)
	}
	return result, nil
}

// GetSecurityEffectiveness returns security effectiveness metrics.
func (c *Client) GetSecurityEffectiveness(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/security/effectiveness?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetSecurityEffectiveness: %w", err)
	}
	return result, nil
}

// GetSecurityReport returns a previously generated security assessment report.
// GET /api/v1/security/report?assessmentId=... — the report store is keyed by
// assessmentId, so the query param MUST be "assessmentId". Sending "scanId"
// 400s ("assessmentId query param is required") on every call.
func (c *Client) GetSecurityReport(ctx context.Context, assessmentID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("assessmentId", assessmentID)
	if err := c.doRequest(ctx, http.MethodGet, "/security/report?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetSecurityReport: %w", err)
	}
	return result, nil
}

// CodeScan scans code for security vulnerabilities.
//
// Fails CLOSED: a 2xx with no `findingsCount`/`findings` is NO VERDICT, not a
// clean build. Both keys assert to nil when absent, so `if len(findings) > 0 {
// fail }` passed a scan that never parsed a line.
func (c *Client) CodeScan(ctx context.Context, code, language, projectID string) (map[string]any, error) {
	body := map[string]any{"code": code, "language": language, "projectId": projectID}
	var verdict CodeScanResult
	result, err := c.postGuardedMap(ctx, "/security/code-scan", body, &verdict)
	if err != nil {
		return nil, fmt.Errorf("CodeScan: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("CodeScan", f, "POST /security/code-scan")
	}
	// Bound to the REQUEST: a scan that parsed the code as a DIFFERENT language
	// than it was written in finds nothing and reports zero findings.
	if reason := verdict.inconsistency(language); reason != "" {
		return nil, uninterpretableVerdict("CodeScan", "POST /security/code-scan", reason)
	}
	return result, nil
}

// --- Traces (extended) ---

// GetTrace returns a specific trace.
func (c *Client) GetTrace(ctx context.Context, traceID string) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/traces/"+traceID, nil, &result); err != nil {
		return nil, fmt.Errorf("GetTrace: %w", err)
	}
	return result, nil
}

// SearchTraces searches traces by query.
func (c *Client) SearchTraces(ctx context.Context, projectID, query string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("q", query)
	if err := c.doRequest(ctx, http.MethodGet, "/traces/search?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("SearchTraces: %w", err)
	}
	return result, nil
}

// CreateTrace creates a new trace.
func (c *Client) CreateTrace(ctx context.Context, projectID, sessionID string, steps []map[string]any) (map[string]any, error) {
	body := map[string]any{"projectId": projectID, "sessionId": sessionID, "steps": steps}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/traces", body, &result); err != nil {
		return nil, fmt.Errorf("CreateTrace: %w", err)
	}
	return result, nil
}

// IngestOTLP ingests OpenTelemetry trace data.
func (c *Client) IngestOTLP(ctx context.Context, resourceSpans []map[string]any) (map[string]any, error) {
	body := map[string]any{"resourceSpans": resourceSpans}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/ingest/otlp/traces", body, &result); err != nil {
		return nil, fmt.Errorf("IngestOTLP: %w", err)
	}
	return result, nil
}

// --- Cost (extended) ---

// GetCostSavings returns cost saving opportunities.
func (c *Client) GetCostSavings(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/cost/savings?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCostSavings: %w", err)
	}
	return result, nil
}

// GetCostBudget returns cost budget configuration.
func (c *Client) GetCostBudget(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/cost/budget?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCostBudget: %w", err)
	}
	return result, nil
}

// GetCostAnomalies returns cost anomaly detection results.
func (c *Client) GetCostAnomalies(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/cost/anomalies?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCostAnomalies: %w", err)
	}
	return result, nil
}

// GetCostRecommendations returns cost optimization recommendations.
func (c *Client) GetCostRecommendations(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/cost/recommendations?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCostRecommendations: %w", err)
	}
	return result, nil
}

// --- Monitoring (extended) ---

// GetMonitoringAnalytics returns monitoring analytics.
func (c *Client) GetMonitoringAnalytics(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/monitoring/analytics?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetMonitoringAnalytics: %w", err)
	}
	return result, nil
}

// GetMonitoringSLA returns SLA monitoring data.
func (c *Client) GetMonitoringSLA(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/monitoring/sla?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetMonitoringSLA: %w", err)
	}
	return result, nil
}

// --- Compliance (extended) ---

// GetCompliance returns the org's compliance assessments (newest first). The
// server replies with a bare array of assessment rows, not an object.
func (c *Client) GetCompliance(ctx context.Context, orgID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/compliance?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetCompliance: %w", err)
	}
	return result, nil
}

// GetComplianceGaps returns compliance gaps.
func (c *Client) GetComplianceGaps(ctx context.Context, orgID, framework string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	q.Set("framework", framework)
	if err := c.doRequest(ctx, http.MethodGet, "/compliance/gaps?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetComplianceGaps: %w", err)
	}
	return result, nil
}

// GetEUAIAct returns EU AI Act compliance status.
func (c *Client) GetEUAIAct(ctx context.Context, orgID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/compliance/eu-ai-act?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetEUAIAct: %w", err)
	}
	return result, nil
}

// GetModelCards returns model cards for compliance.
func (c *Client) GetModelCards(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/compliance/model-cards?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetModelCards: %w", err)
	}
	return result, nil
}

// --- Datasets (extended) ---

// ListDatasets returns all datasets for a project.
func (c *Client) ListDatasets(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/datasets?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListDatasets: %w", err)
	}
	return result, nil
}

// --- Annotations ---

// AnnotationLabels are the valid values for CreateAnnotation's label argument.
// POST /api/v1/annotations validates label against this exact enum and 400s on
// anything else. The server derives a default score from the label when none is
// given: good → 1.0, bad → 0.0, unsure → 0.5.
var AnnotationLabels = []string{"good", "bad", "unsure"}

// CreateAnnotation creates an annotation on a log entry.
//
// label MUST be one of AnnotationLabels ("good", "bad", or "unsure") — the
// server rejects any other value with 400. It is checked client-side here so
// callers get an actionable error before the wire instead of an opaque 400.
func (c *Client) CreateAnnotation(ctx context.Context, projectID, logID, label string) (map[string]any, error) {
	valid := false
	for _, l := range AnnotationLabels {
		if label == l {
			valid = true
			break
		}
	}
	if !valid {
		return nil, &EvalGuardError{
			Code:    ErrCodeValidation,
			Message: fmt.Sprintf("CreateAnnotation: label must be one of %v, got %q", AnnotationLabels, label),
		}
	}
	body := map[string]any{"projectId": projectID, "logId": logID, "label": label}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/annotations", body, &result); err != nil {
		return nil, fmt.Errorf("CreateAnnotation: %w", err)
	}
	return result, nil
}

// ListAnnotations returns annotations for a project.
func (c *Client) ListAnnotations(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/annotations?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListAnnotations: %w", err)
	}
	return result, nil
}

// --- Webhooks ---

// ListWebhooks returns webhooks for an organization.
func (c *Client) ListWebhooks(ctx context.Context, orgID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/webhooks?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListWebhooks: %w", err)
	}
	return result, nil
}

// ListApiKeys returns API keys for an organization.
func (c *Client) ListApiKeys(ctx context.Context, orgID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/api-keys?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListApiKeys: %w", err)
	}
	return result, nil
}

// --- Remaining parity methods ---

// GetSIEMConnectors returns SIEM connector configuration.
func (c *Client) GetSIEMConnectors(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/siem?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetSIEMConnectors: %w", err)
	}
	return result, nil
}

// GetSettings returns project settings.
func (c *Client) GetSettings(ctx context.Context, projectID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/settings?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetSettings: %w", err)
	}
	return result, nil
}

// ListNotifications returns the current user's notifications plus the unread
// count. The server replies with an object {notifications, unread_count}.
func (c *Client) ListNotifications(ctx context.Context) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/notifications", nil, &result); err != nil {
		return nil, fmt.Errorf("ListNotifications: %w", err)
	}
	return result, nil
}

// TemplatesResponse is the data payload of GET /api/v1/templates — the API
// returns apiSuccess({ templates, count }), NOT a bare array. The previous
// []map[string]any target unmarshalled the {templates,count} OBJECT into a
// slice (a no-op), so callers always got nil. Templates holds the rows; Count
// is the number of templates returned.
type TemplatesResponse struct {
	Templates []map[string]any `json:"templates"`
	Count     int              `json:"count"`
}

// ListTemplates returns available eval templates.
func (c *Client) ListTemplates(ctx context.Context) (*TemplatesResponse, error) {
	var result TemplatesResponse
	if err := c.doRequest(ctx, http.MethodGet, "/templates", nil, &result); err != nil {
		return nil, fmt.Errorf("ListTemplates: %w", err)
	}
	return &result, nil
}

// GetMarketplace returns the public marketplace template catalog.
// GET /api/v1/marketplace replies with a JSON ARRAY of template rows
// (apiSuccess(enriched) where enriched is an array), so the result must decode
// into a slice — a map[string]any target fails with "cannot unmarshal array
// into Go value of type map[string]interface{}".
func (c *Client) GetMarketplace(ctx context.Context) ([]map[string]any, error) {
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/marketplace", nil, &result); err != nil {
		return nil, fmt.Errorf("GetMarketplace: %w", err)
	}
	return result, nil
}

// ListEvalSchedules returns eval schedules.
func (c *Client) ListEvalSchedules(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/eval-schedules?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListEvalSchedules: %w", err)
	}
	return result, nil
}

// ListIncidents returns incidents for a project.
func (c *Client) ListIncidents(ctx context.Context, projectID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	if err := c.doRequest(ctx, http.MethodGet, "/incidents?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListIncidents: %w", err)
	}
	return result, nil
}

// GetDashboardStats returns dashboard overview stats.
func (c *Client) GetDashboardStats(ctx context.Context) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/dashboard/stats", nil, &result); err != nil {
		return nil, fmt.Errorf("GetDashboardStats: %w", err)
	}
	return result, nil
}

// DetectDrift compares two eval runs for drift. POST /api/v1/monitoring/drift/detect.
func (c *Client) DetectDrift(ctx context.Context, baselineRunID, currentRunID string) (map[string]any, error) {
	body := map[string]any{"baselineRunId": baselineRunID, "currentRunId": currentRunID}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/monitoring/drift/detect", body, &result); err != nil {
		return nil, fmt.Errorf("DetectDrift: %w", err)
	}
	return result, nil
}

// SmartRoute routes test cases to appropriate model tiers.
func (c *Client) SmartRoute(ctx context.Context, testCases []map[string]any) (map[string]any, error) {
	body := map[string]any{"testCases": testCases}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/smart-routing/test-cases", body, &result); err != nil {
		return nil, fmt.Errorf("SmartRoute: %w", err)
	}
	return result, nil
}

// GetAutopilotConfig returns autopilot configuration.
func (c *Client) GetAutopilotConfig(ctx context.Context) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/autopilot", nil, &result); err != nil {
		return nil, fmt.Errorf("GetAutopilotConfig: %w", err)
	}
	return result, nil
}

// ListPipelines returns pipeline templates and the caller's custom pipelines.
// The server replies with an object {templates, custom}, not a bare array.
func (c *Client) ListPipelines(ctx context.Context) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/pipelines", nil, &result); err != nil {
		return nil, fmt.Errorf("ListPipelines: %w", err)
	}
	return result, nil
}

// GenerateGuardrails generates guardrails from description.
func (c *Client) GenerateGuardrails(ctx context.Context, description, projectID string) (map[string]any, error) {
	body := map[string]any{"description": description, "projectId": projectID}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/guardrails/generate", body, &result); err != nil {
		return nil, fmt.Errorf("GenerateGuardrails: %w", err)
	}
	return result, nil
}

// AISBOMAgent describes an AI agent to fold into a generated AI-SBOM. The
// server's scanProject does not auto-detect agents, so pass any that were
// discovered (e.g. from agent-connector discovery) here. Name and Source are
// required by POST /ai-sbom/generate; the rest are optional.
type AISBOMAgent struct {
	Name       string   `json:"name"`
	Model      string   `json:"model,omitempty"`
	Provider   string   `json:"provider,omitempty"`
	Tools      []string `json:"tools,omitempty"`
	Guardrails []string `json:"guardrails,omitempty"`
	Source     string   `json:"source"`
}

// GenerateAISBOMOptions carries the optional manifest and scan inputs for
// GenerateAISBOM. Every field is optional — the more manifests / lockfiles you
// supply, the deeper the supply-chain (CVE + typosquat) coverage over the
// resolved dependency graph. Mirrors the optional half of the
// POST /ai-sbom/generate body (see the route's PostBody) and the TS/Python SDKs;
// projectName, the sole required field, is GenerateAISBOM's first argument.
type GenerateAISBOMOptions struct {
	// ProjectVersion labels the generated BOM (defaults to "1.0.0" server-side).
	ProjectVersion string `json:"projectVersion,omitempty"`
	// Format selects the export: "json" (default), "cyclonedx", or "spdx".
	Format string `json:"format,omitempty"`
	// PackageJSON / PackageLockJSON are parsed npm manifests (arbitrary JSON).
	PackageJSON     map[string]any `json:"packageJson,omitempty"`
	PackageLockJSON map[string]any `json:"packageLockJson,omitempty"`
	// PythonRequirements / PoetryLock are the raw text of the respective files.
	PythonRequirements string `json:"pythonRequirements,omitempty"`
	PoetryLock         string `json:"poetryLock,omitempty"`
	// GoSum / GoMod are the raw text of go.sum / go.mod (transitive coverage).
	GoSum string `json:"goSum,omitempty"`
	GoMod string `json:"goMod,omitempty"`
	// PomXML / BuildGradle / GradleLockfile are the raw text of the JVM manifests.
	PomXML         string `json:"pomXml,omitempty"`
	BuildGradle    string `json:"buildGradle,omitempty"`
	GradleLockfile string `json:"gradleLockfile,omitempty"`
	// EvalguardConfig / PromptRegistry / DatasetRegistry are arbitrary JSON that
	// let the generator surface EvalGuard-managed resources in the BOM.
	EvalguardConfig map[string]any `json:"evalguardConfig,omitempty"`
	PromptRegistry  map[string]any `json:"promptRegistry,omitempty"`
	DatasetRegistry map[string]any `json:"datasetRegistry,omitempty"`
	// ProviderKeys are provider identifiers (not secrets) to fold in.
	ProviderKeys []string `json:"providerKeys,omitempty"`
	// Agents are AI agents to include (scanProject does not detect them).
	Agents []AISBOMAgent `json:"agents,omitempty"`
	// LiveCveScan toggles live OSV.dev lookups. It is ON by default server-side;
	// set it to false (via a pointer to false) for an offline-only scan. Left nil
	// the field is omitted and the server default (on) applies.
	LiveCveScan *bool `json:"liveCveScan,omitempty"`
}

// GenerateAISBOM auto-generates an AI Software Bill of Materials from project
// manifests, with a supply-chain scan: live OSV.dev CVE lookups (default on)
// over the full resolved dependency graph, an embedded offline CVE database, and
// typosquat detection.
//
// projectName is REQUIRED — POST /ai-sbom/generate validates on projectName, not
// projectId. The previous {projectId} body was rejected on every call with
// 400 "projectName: Invalid input"; this now sends {projectName, ...opts}.
// Unlike GetAISBOM this is NOT project-id scoped: the BOM is built from the
// manifests you supply, keyed by projectName. Pass lockfiles/manifests via opts
// for deeper transitive coverage; opts may be nil for a bare scan.
func (c *Client) GenerateAISBOM(ctx context.Context, projectName string, opts *GenerateAISBOMOptions) (map[string]any, error) {
	if strings.TrimSpace(projectName) == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "GenerateAISBOM: projectName is required"}
	}
	if opts == nil {
		opts = &GenerateAISBOMOptions{}
	}
	// Embed the options so their json tags (with omitempty) are promoted onto the
	// wire body alongside the always-present projectName.
	body := struct {
		ProjectName string `json:"projectName"`
		*GenerateAISBOMOptions
	}{ProjectName: projectName, GenerateAISBOMOptions: opts}

	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/ai-sbom/generate", body, &result); err != nil {
		return nil, fmt.Errorf("GenerateAISBOM: %w", err)
	}
	return result, nil
}

// Search performs a full-text search.
func (c *Client) Search(ctx context.Context, projectID, query string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("q", query)
	if err := c.doRequest(ctx, http.MethodGet, "/search?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("Search: %w", err)
	}
	return result, nil
}

// ListTickets returns support tickets.
func (c *Client) ListTickets(ctx context.Context) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/support", nil, &result); err != nil {
		return nil, fmt.Errorf("ListTickets: %w", err)
	}
	return result, nil
}

// ─── Provider Keys (BYOK vault) ───────────────────────────────────────────
//
// Plaintext API keys are encrypted server-side via Supabase Vault envelope
// encryption; responses never include the plaintext. See the SaaS docs at
// https://evalguard.ai/docs/api#provider-keys for the security model.

// ProviderKey is the safe metadata view of a stored provider key — never
// contains plaintext or ciphertext. For identification, use KeyLast4.
type ProviderKey struct {
	ID        string  `json:"id"`
	Provider  string  `json:"provider"`
	ProjectID *string `json:"project_id,omitempty"`
	Label     *string `json:"label,omitempty"`
	KeyLast4  *string `json:"key_last4,omitempty"`
	CreatedAt string  `json:"created_at"`
	RotatedAt *string `json:"rotated_at,omitempty"`
}

type ListProviderKeysResponse struct {
	Keys  []ProviderKey `json:"keys"`
	Total int           `json:"total"`
}

type UpsertProviderKeyRequest struct {
	OrgID     string  `json:"orgId"`
	Provider  string  `json:"provider"`
	APIKey    string  `json:"apiKey"`
	ProjectID *string `json:"projectId,omitempty"`
	Label     *string `json:"label,omitempty"`
}

type UpsertProviderKeyResponse struct {
	Key     ProviderKey `json:"key"`
	Rotated bool        `json:"rotated"`
}

// ListProviderKeys fetches metadata for all BYOK keys in the given org.
func (c *Client) ListProviderKeys(ctx context.Context, orgID string, projectID *string) (*ListProviderKeysResponse, error) {
	q := url.Values{}
	q.Set("orgId", orgID)
	if projectID != nil {
		q.Set("projectId", *projectID)
	}
	var result ListProviderKeysResponse
	if err := c.doRequest(ctx, http.MethodGet, "/provider-keys?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListProviderKeys: %w", err)
	}
	return &result, nil
}

// UpsertProviderKey creates a new BYOK key or rotates an existing one (if a
// row already exists for the (org, project, provider) triple). The returned
// `Rotated` flag indicates which path was taken.
func (c *Client) UpsertProviderKey(ctx context.Context, req UpsertProviderKeyRequest) (*UpsertProviderKeyResponse, error) {
	var result UpsertProviderKeyResponse
	if err := c.doRequest(ctx, http.MethodPost, "/provider-keys", req, &result); err != nil {
		return nil, fmt.Errorf("UpsertProviderKey: %w", err)
	}
	return &result, nil
}

// DeleteProviderKey revokes a BYOK key. The underlying vault.secrets row is
// auto-cleaned by the DB trigger installed in migration 20260424.
func (c *Client) DeleteProviderKey(ctx context.Context, orgID, keyID string) error {
	q := url.Values{}
	q.Set("orgId", orgID)
	q.Set("id", keyID)
	if err := c.doRequest(ctx, http.MethodDelete, "/provider-keys?"+q.Encode(), nil, nil); err != nil {
		return fmt.Errorf("DeleteProviderKey: %w", err)
	}
	return nil
}

// ─── Models Registry (custom pricing overrides) ───────────────────────────

type ModelRegistryEntry struct {
	ID                  string  `json:"id"`
	ModelName           string  `json:"model_name"`
	Provider            *string `json:"provider,omitempty"`
	DisplayName         *string `json:"display_name,omitempty"`
	InputPricePer1MUSD  float64 `json:"input_price_per_1m_usd"`
	OutputPricePer1MUSD float64 `json:"output_price_per_1m_usd"`
	ContextWindow       *int    `json:"context_window,omitempty"`
	Notes               *string `json:"notes,omitempty"`
	ProjectID           *string `json:"project_id,omitempty"`
}

type ListModelsResponse struct {
	Models []ModelRegistryEntry `json:"models"`
	Total  int                  `json:"total"`
}

type UpsertModelRequest struct {
	OrgID               string  `json:"orgId"`
	ModelName           string  `json:"modelName"`
	InputPricePer1MUSD  float64 `json:"inputPricePer1mUsd"`
	OutputPricePer1MUSD float64 `json:"outputPricePer1mUsd"`
	ProjectID           *string `json:"projectId,omitempty"`
	Provider            *string `json:"provider,omitempty"`
	DisplayName         *string `json:"displayName,omitempty"`
	ContextWindow       *int    `json:"contextWindow,omitempty"`
	Notes               *string `json:"notes,omitempty"`
}

// ListModels returns custom pricing overrides for the org (project-specific
// + org-default rows interleaved).
func (c *Client) ListModels(ctx context.Context, orgID string, projectID *string) (*ListModelsResponse, error) {
	q := url.Values{}
	q.Set("orgId", orgID)
	if projectID != nil {
		q.Set("projectId", *projectID)
	}
	var result ListModelsResponse
	if err := c.doRequest(ctx, http.MethodGet, "/models/registry?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListModels: %w", err)
	}
	return &result, nil
}

// UpsertModel creates or updates a pricing override. Prices are in USD per
// million tokens. Invalidates the server-side cost cache so new pricing
// takes effect within 60 s.
func (c *Client) UpsertModel(ctx context.Context, req UpsertModelRequest) (*ModelRegistryEntry, error) {
	var result struct {
		Model   ModelRegistryEntry `json:"model"`
		Created bool               `json:"created"`
	}
	if err := c.doRequest(ctx, http.MethodPost, "/models/registry", req, &result); err != nil {
		return nil, fmt.Errorf("UpsertModel: %w", err)
	}
	return &result.Model, nil
}

func (c *Client) DeleteModel(ctx context.Context, orgID, modelID string) error {
	q := url.Values{}
	q.Set("orgId", orgID)
	q.Set("id", modelID)
	if err := c.doRequest(ctx, http.MethodDelete, "/models/registry?"+q.Encode(), nil, nil); err != nil {
		return fmt.Errorf("DeleteModel: %w", err)
	}
	return nil
}

// ─── API-key budget caps ──────────────────────────────────────────────────

type APIKeyBudget struct {
	KeyID                  string   `json:"keyId"`
	Name                   string   `json:"name,omitempty"`
	MonthlyBudgetUSD       *float64 `json:"monthlyBudgetUsd"`
	CurrentPeriodSpentUSD  float64  `json:"currentPeriodSpentUsd"`
	CurrentPeriodStartedAt string   `json:"currentPeriodStartedAt"`
	RemainingUSD           *float64 `json:"remainingUsd,omitempty"`
	PercentUsed            *float64 `json:"percentUsed,omitempty"`
	StaleReset             bool     `json:"staleReset,omitempty"`
}

// GetAPIKeyBudget returns the current month's spend + cap + percentUsed for
// a virtual API key. `staleReset: true` means a month rollover is pending
// and the next gateway request will reset the counter to 0.
func (c *Client) GetAPIKeyBudget(ctx context.Context, keyID string) (*APIKeyBudget, error) {
	var result APIKeyBudget
	path := fmt.Sprintf("/api-keys/%s/budget", url.PathEscape(keyID))
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("GetAPIKeyBudget: %w", err)
	}
	return &result, nil
}

// SetAPIKeyBudget updates the monthly USD cap. Pass nil to remove the cap.
// Returns 402 Payment Required from the gateway proxy once spend reaches
// the cap (see /api/v1/gateway/proxy enforcement).
func (c *Client) SetAPIKeyBudget(ctx context.Context, keyID string, monthlyBudgetUSD *float64) (*APIKeyBudget, error) {
	body := map[string]any{"monthlyBudgetUsd": monthlyBudgetUSD}
	var result APIKeyBudget
	path := fmt.Sprintf("/api-keys/%s/budget", url.PathEscape(keyID))
	if err := c.doRequest(ctx, http.MethodPatch, path, body, &result); err != nil {
		return nil, fmt.Errorf("SetAPIKeyBudget: %w", err)
	}
	return &result, nil
}

func (c *Client) RemoveAPIKeyBudget(ctx context.Context, keyID string) error {
	path := fmt.Sprintf("/api-keys/%s/budget", url.PathEscape(keyID))
	if err := c.doRequest(ctx, http.MethodDelete, path, nil, nil); err != nil {
		return fmt.Errorf("RemoveAPIKeyBudget: %w", err)
	}
	return nil
}

// ─── Trace attachments (inline blob storage) ──────────────────────────────

type SpanAttachment struct {
	ID        string         `json:"id"`
	SpanID    string         `json:"span_id"`
	Name      string         `json:"name"`
	MimeType  string         `json:"mime_type"`
	SizeBytes int            `json:"size_bytes"`
	Metadata  map[string]any `json:"metadata"`
	CreatedAt string         `json:"created_at"`
}

type ListAttachmentsResponse struct {
	Attachments []SpanAttachment `json:"attachments"`
	Total       int              `json:"total"`
}

type UploadAttachmentRequest struct {
	TraceID   string `json:"-"`
	ProjectID string `json:"projectId"`
	SpanID    string `json:"spanId"`
	Name      string `json:"name"`
	MimeType  string `json:"mimeType"`
	// Data is the raw bytes to attach. Encoded to base64 on the wire.
	Data []byte `json:"-"`
	// DataBase64 is populated at send-time; clients usually leave it empty.
	DataBase64 string         `json:"dataBase64,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

const maxAttachmentBytes = 1 << 20 // 1 MB — matches server-side CHECK constraint

// ListTraceAttachments returns metadata for all attachments on the given
// trace. Binary payload is fetched per-attachment via FetchTraceAttachment.
func (c *Client) ListTraceAttachments(ctx context.Context, traceID, projectID string) (*ListAttachmentsResponse, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	var result ListAttachmentsResponse
	path := fmt.Sprintf("/traces/%s/attachments?%s", url.PathEscape(traceID), q.Encode())
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("ListTraceAttachments: %w", err)
	}
	return &result, nil
}

// UploadTraceAttachment pushes a binary blob (image/audio/text/json/pdf) to
// a specific span. Enforces the 1 MB limit client-side to avoid a wasted
// round trip.
func (c *Client) UploadTraceAttachment(ctx context.Context, req UploadAttachmentRequest) (*SpanAttachment, error) {
	if len(req.Data) == 0 && req.DataBase64 == "" {
		return nil, fmt.Errorf("UploadTraceAttachment: Data or DataBase64 is required")
	}
	if len(req.Data) > maxAttachmentBytes {
		return nil, fmt.Errorf("UploadTraceAttachment: payload exceeds 1 MB (%d bytes)", len(req.Data))
	}
	if req.DataBase64 == "" && len(req.Data) > 0 {
		req.DataBase64 = base64.StdEncoding.EncodeToString(req.Data)
	}

	var result struct {
		Attachment SpanAttachment `json:"attachment"`
	}
	path := fmt.Sprintf("/traces/%s/attachments", url.PathEscape(req.TraceID))
	if err := c.doRequest(ctx, http.MethodPost, path, req, &result); err != nil {
		return nil, fmt.Errorf("UploadTraceAttachment: %w", err)
	}
	return &result.Attachment, nil
}

// FetchTraceAttachment downloads the raw bytes of an attachment.
// The returned Content-Type corresponds to the stored mime_type.
//
// Like every sibling method, this goes through the shared retry/backoff loop
// (doRaw), so a transient 429/5xx is retried with Retry-After honored, and a
// failure surfaces as the typed *EvalGuardError the rest of the SDK returns.
// It previously hand-rolled its own request — skipping the retry loop and
// returning an unstructured fmt.Errorf("HTTP %d") string that callers could
// not errors.As into an *EvalGuardError / *AuthError / *RateLimitError.
func (c *Client) FetchTraceAttachment(ctx context.Context, traceID, attachmentID, projectID string) ([]byte, string, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	path := fmt.Sprintf("/traces/%s/attachments/%s?%s", url.PathEscape(traceID), url.PathEscape(attachmentID), q.Encode())
	respBody, header, err := c.doRaw(ctx, http.MethodGet, path, "application/octet-stream", nil)
	if err != nil {
		return nil, "", fmt.Errorf("FetchTraceAttachment: %w", err)
	}
	return respBody, header.Get("Content-Type"), nil
}

func (c *Client) DeleteTraceAttachment(ctx context.Context, traceID, attachmentID, projectID string) error {
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("id", attachmentID)
	path := fmt.Sprintf("/traces/%s/attachments?%s", url.PathEscape(traceID), q.Encode())
	if err := c.doRequest(ctx, http.MethodDelete, path, nil, nil); err != nil {
		return fmt.Errorf("DeleteTraceAttachment: %w", err)
	}
	return nil
}

// --- Agent-run metered billing (Gap #5) ---

type StartAgentRunOpts struct {
	APIKeyID      string         `json:"apiKeyId,omitempty"`
	EndCustomerID string         `json:"endCustomerId,omitempty"`
	TraceID       string         `json:"traceId,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type AgentRun struct {
	RunID     string `json:"runId"`
	Status    string `json:"status"`
	StartedAt string `json:"startedAt"`
}

func (c *Client) StartAgentRun(ctx context.Context, opts StartAgentRunOpts) (*AgentRun, error) {
	var result AgentRun
	if err := c.doRequest(ctx, http.MethodPost, "/agent-runs/start", opts, &result); err != nil {
		return nil, fmt.Errorf("StartAgentRun: %w", err)
	}
	return &result, nil
}

type EndAgentRunOpts struct {
	CostUSD   float64        `json:"costUsd"`
	TokensIn  int            `json:"tokensIn,omitempty"`
	TokensOut int            `json:"tokensOut,omitempty"`
	Status    string         `json:"status,omitempty"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

func (c *Client) EndAgentRun(ctx context.Context, runID string, opts EndAgentRunOpts) error {
	path := fmt.Sprintf("/agent-runs/%s/end", url.PathEscape(runID))
	if err := c.doRequest(ctx, http.MethodPost, path, opts, nil); err != nil {
		return fmt.Errorf("EndAgentRun: %w", err)
	}
	return nil
}

type ListAgentRunsOpts struct {
	APIKeyID      string
	AgentTag      string
	EndCustomerID string
	Since         string
	Limit         int
	GroupBy       string // "agent_tag" | "end_customer_id" | "api_key_id"
}

func (c *Client) ListAgentRuns(ctx context.Context, opts ListAgentRunsOpts) (map[string]any, error) {
	q := url.Values{}
	if opts.APIKeyID != "" {
		q.Set("apiKeyId", opts.APIKeyID)
	}
	if opts.AgentTag != "" {
		q.Set("agentTag", opts.AgentTag)
	}
	if opts.EndCustomerID != "" {
		q.Set("endCustomerId", opts.EndCustomerID)
	}
	if opts.Since != "" {
		q.Set("since", opts.Since)
	}
	if opts.Limit > 0 {
		q.Set("limit", fmt.Sprintf("%d", opts.Limit))
	}
	if opts.GroupBy != "" {
		q.Set("groupBy", opts.GroupBy)
	}
	path := "/agent-runs"
	if q.Encode() != "" {
		path += "?" + q.Encode()
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("ListAgentRuns: %w", err)
	}
	return result, nil
}

// --- Model-scan governance (Gap #1) ---

type PromoteModelScanOpts struct {
	ToEnv    string `json:"toEnv"`
	FromEnv  string `json:"fromEnv,omitempty"`
	Override bool   `json:"override,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

func (c *Client) PromoteModelScan(ctx context.Context, scanID string, opts PromoteModelScanOpts) (map[string]any, error) {
	path := fmt.Sprintf("/security/model-scan/%s/promote", url.PathEscape(scanID))
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, path, opts, &result); err != nil {
		return nil, fmt.Errorf("PromoteModelScan: %w", err)
	}
	return result, nil
}

// GetModelScanAttestation returns the CycloneDX-ML 1.6 attestation JSON for a scan.
func (c *Client) GetModelScanAttestation(ctx context.Context, scanID string) (map[string]any, error) {
	path := fmt.Sprintf("/security/model-scan/%s/attestation", url.PathEscape(scanID))
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("GetModelScanAttestation: %w", err)
	}
	return result, nil
}

// --- Shadow-AI discovery (Gap #2) ---

func (c *Client) IngestShadowAISightings(ctx context.Context, source string, rows []map[string]any, projectID string) (map[string]any, error) {
	body := map[string]any{"source": source, "rows": rows}
	if projectID != "" {
		body["projectId"] = projectID
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/shadow-ai/ingest", body, &result); err != nil {
		return nil, fmt.Errorf("IngestShadowAISightings: %w", err)
	}
	return result, nil
}

func (c *Client) SetShadowAIPolicy(ctx context.Context, domain, status, rationale, projectID string) error {
	body := map[string]any{"domain": domain, "status": status}
	if rationale != "" {
		body["rationale"] = rationale
	}
	if projectID != "" {
		body["projectId"] = projectID
	}
	if err := c.doRequest(ctx, http.MethodPost, "/shadow-ai/policy", body, nil); err != nil {
		return fmt.Errorf("SetShadowAIPolicy: %w", err)
	}
	return nil
}

// --- SIEM inbound tokens (Gap #6) ---

type CreateSiemInboundTokenOpts struct {
	Source          string   `json:"source"` // splunk | sentinel | qradar | generic_webhook
	Label           string   `json:"label"`
	AllowedActions  []string `json:"allowedActions,omitempty"`
	RateLimitPerMin int      `json:"rateLimitPerMin,omitempty"`
	ProjectID       string   `json:"projectId,omitempty"`
}

// CreateSiemInboundToken mints a SIEM webhook HMAC token. The returned hmacSecret
// in the "data.token" map is shown EXACTLY ONCE — save it into your SIEM now.
func (c *Client) CreateSiemInboundToken(ctx context.Context, opts CreateSiemInboundTokenOpts) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/siem/inbound/tokens", opts, &result); err != nil {
		return nil, fmt.Errorf("CreateSiemInboundToken: %w", err)
	}
	return result, nil
}

func (c *Client) RevokeSiemInboundToken(ctx context.Context, tokenID, projectID string) error {
	q := url.Values{}
	q.Set("id", tokenID)
	q.Set("projectId", projectID)
	path := "/siem/inbound/tokens?" + q.Encode()
	if err := c.doRequest(ctx, http.MethodDelete, path, nil, nil); err != nil {
		return fmt.Errorf("RevokeSiemInboundToken: %w", err)
	}
	return nil
}

// --- Debug agent (Gap #4) ---

type AnalyzeTraceOpts struct {
	TraceID          string         `json:"traceId"`
	ScorerResultIDs  []string       `json:"scorerResultIds,omitempty"`
	AnalyzerModel    string         `json:"analyzerModel,omitempty"`
	AnalyzerProvider string         `json:"analyzerProvider,omitempty"`
	ExpectedOutput   string         `json:"expectedOutput,omitempty"`
	InlineContext    map[string]any `json:"inlineContext,omitempty"`
	ProjectID        string         `json:"projectId,omitempty"`
}

// AnalyzeTrace asks the debug agent to analyze a failing trace. Returns a map
// with sessionId, fixKind, confidence, rationale, suggestedFix, analyzerCostUsd.
func (c *Client) AnalyzeTrace(ctx context.Context, opts AnalyzeTraceOpts) (map[string]any, error) {
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/debug-agent", opts, &result); err != nil {
		return nil, fmt.Errorf("AnalyzeTrace: %w", err)
	}
	return result, nil
}

// ─── Agent tools (the agent-builder tool registry) ────────────────────────
//
// CRUD + a dry-run test harness for the tools an agent workflow can call.
// A tool is one of three kinds — a REST call, a sandboxed code snippet, or an
// MCP server invocation — described by a JSON-Schema parameter object so the
// builder UI can render an input form. Routes live under /api/v1/agent-tools;
// every call is project-scoped (projectId, a UUID, is required server-side).

// AgentToolParameters is the JSON-Schema object describing a tool's inputs.
// Type is always "object"; Properties maps each argument name to its schema
// fragment; Required lists the mandatory argument names.
type AgentToolParameters struct {
	Type       string         `json:"type"`
	Properties map[string]any `json:"properties"`
	Required   []string       `json:"required,omitempty"`
}

// AgentToolREST configures a "rest" tool — an outbound HTTP request the agent
// makes when the tool is invoked. BodyTemplate may interpolate the tool's
// arguments; Auth, when set, injects a credential header server-side (the
// plaintext value is never echoed back — see AgentTool.HasSecret).
type AgentToolREST struct {
	Method       string            `json:"method"`
	URL          string            `json:"url"`
	Headers      map[string]string `json:"headers,omitempty"`
	Auth         *AgentToolAuth    `json:"auth,omitempty"`
	BodyTemplate string            `json:"bodyTemplate,omitempty"`
	TimeoutMs    int               `json:"timeoutMs,omitempty"`
}

// AgentToolAuth describes how a "rest" tool authenticates. Type is e.g.
// "bearer" or "header"; Header names the header to set when Type is "header";
// Value is the secret credential (write-only — responses omit it).
type AgentToolAuth struct {
	Type   string `json:"type"`
	Header string `json:"header,omitempty"`
	Value  string `json:"value,omitempty"`
}

// AgentToolCode configures a "code" tool — a sandboxed snippet evaluated with
// the tool arguments in scope. TimeoutMs caps the execution wall clock.
type AgentToolCode struct {
	Source    string `json:"source"`
	TimeoutMs int    `json:"timeoutMs,omitempty"`
}

// AgentToolMCP configures an "mcp" tool — a call to a named tool on a Model
// Context Protocol server. ToolName defaults to the AgentTool name when empty.
type AgentToolMCP struct {
	Server   string `json:"server"`
	ToolName string `json:"toolName,omitempty"`
}

// AgentTool is a single tool the agent-builder can wire into a workflow. ID is
// assigned by the server on create and echoed on reads. Type selects which of
// REST/Code/MCP is populated. HasSecret is a server-set read-only flag that is
// true when a credential is stored for the tool (the plaintext is never
// returned). Mirrors the AgentTool shape on /api/v1/agent-tools.
type AgentTool struct {
	ID          string              `json:"id,omitempty"`
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	Type        string              `json:"type"` // "rest" | "code" | "mcp"
	Parameters  AgentToolParameters `json:"parameters"`
	REST        *AgentToolREST      `json:"rest,omitempty"`
	Code        *AgentToolCode      `json:"code,omitempty"`
	MCP         *AgentToolMCP       `json:"mcp,omitempty"`
	HasSecret   bool                `json:"hasSecret,omitempty"`
}

// listAgentToolsResponse is the GET /agent-tools envelope's data object.
type listAgentToolsResponse struct {
	Tools []AgentTool `json:"tools"`
}

// agentToolBody is the create/update request body: a project scope plus the
// tool definition. PATCH and POST share this shape.
type agentToolBody struct {
	ProjectID string    `json:"projectId"`
	Tool      AgentTool `json:"tool"`
}

// AgentToolTestResult is the outcome of POST /agent-tools/{id}/test — a dry
// run of the tool with caller-supplied arguments. Ok is the headline verdict;
// Stage names where execution got to (e.g. "validate", "request", "response");
// Status is the upstream HTTP status for a "rest" tool when one was reached;
// Body is the captured upstream payload; Issues lists per-argument validation
// problems; Message is a human-readable summary.
type AgentToolTestResult struct {
	Ok      bool     `json:"ok"`
	Stage   string   `json:"stage"`
	Status  int      `json:"status,omitempty"`
	Body    any      `json:"body,omitempty"`
	Issues  []string `json:"issues,omitempty"`
	Message string   `json:"message,omitempty"`
}

// CreateAgentTool registers a new agent tool in a project.
//
// POST /api/v1/agent-tools with { projectId, tool } returns 201 with the
// stored tool (server-assigned ID, HasSecret reflecting any credential).
func (c *Client) CreateAgentTool(ctx context.Context, projectID string, tool AgentTool) (*AgentTool, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "CreateAgentTool: projectID is required"}
	}
	var result AgentTool
	body := agentToolBody{ProjectID: projectID, Tool: tool}
	if err := c.doRequest(ctx, http.MethodPost, "/agent-tools", body, &result); err != nil {
		return nil, fmt.Errorf("CreateAgentTool: %w", err)
	}
	return &result, nil
}

// GetAgentTool fetches a single agent tool by ID. The credential plaintext is
// never returned; HasSecret indicates whether one is stored.
// GET /api/v1/agent-tools/{id}?projectId=...
func (c *Client) GetAgentTool(ctx context.Context, toolID, projectID string) (*AgentTool, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "GetAgentTool: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	var result AgentTool
	path := fmt.Sprintf("/agent-tools/%s?%s", url.PathEscape(toolID), q.Encode())
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("GetAgentTool: %w", err)
	}
	return &result, nil
}

// ListAgentTools returns all agent tools for a project.
// GET /api/v1/agent-tools?projectId=... — projectId is required.
func (c *Client) ListAgentTools(ctx context.Context, projectID string) ([]AgentTool, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ListAgentTools: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	var result listAgentToolsResponse
	if err := c.doRequest(ctx, http.MethodGet, "/agent-tools?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListAgentTools: %w", err)
	}
	return result.Tools, nil
}

// UpdateAgentTool patches an existing agent tool. PATCH /api/v1/agent-tools/{id}
// with { projectId, tool } returns the stored tool after the merge.
func (c *Client) UpdateAgentTool(ctx context.Context, toolID, projectID string, tool AgentTool) (*AgentTool, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "UpdateAgentTool: projectID is required"}
	}
	var result AgentTool
	body := agentToolBody{ProjectID: projectID, Tool: tool}
	path := fmt.Sprintf("/agent-tools/%s", url.PathEscape(toolID))
	if err := c.doRequest(ctx, http.MethodPatch, path, body, &result); err != nil {
		return nil, fmt.Errorf("UpdateAgentTool: %w", err)
	}
	return &result, nil
}

// DeleteAgentTool removes an agent tool. DELETE /api/v1/agent-tools/{id}?projectId=...
// Returns the deleted id; an error otherwise.
func (c *Client) DeleteAgentTool(ctx context.Context, toolID, projectID string) error {
	if projectID == "" {
		return &EvalGuardError{Code: ErrCodeValidation, Message: "DeleteAgentTool: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	path := fmt.Sprintf("/agent-tools/%s?%s", url.PathEscape(toolID), q.Encode())
	if err := c.doRequest(ctx, http.MethodDelete, path, nil, nil); err != nil {
		return fmt.Errorf("DeleteAgentTool: %w", err)
	}
	return nil
}

// TestAgentTool dry-runs a tool with the given arguments and returns the
// execution outcome. POST /api/v1/agent-tools/{id}/test with { projectId, args }.
func (c *Client) TestAgentTool(ctx context.Context, toolID, projectID string, args map[string]any) (*AgentToolTestResult, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "TestAgentTool: projectID is required"}
	}
	body := map[string]any{"projectId": projectID, "args": args}
	var result AgentToolTestResult
	path := fmt.Sprintf("/agent-tools/%s/test", url.PathEscape(toolID))
	if err := c.doRequest(ctx, http.MethodPost, path, body, &result); err != nil {
		return nil, fmt.Errorf("TestAgentTool: %w", err)
	}
	return &result, nil
}

// ─── Abuse reports (defense-in-depth intake) ──────────────────────────────
//
// Trust-and-safety intake: a reporter flags a subject under a category, and
// the server returns the stored report plus an auto-triage verdict (severity,
// dedup key, escalation/detector-feed flags). Routes are project-scoped under
// /api/v1/abuse-reports.

// AbuseReport is a stored trust-and-safety report.
type AbuseReport struct {
	ID          string         `json:"id"`
	ProjectID   string         `json:"projectId,omitempty"`
	Category    string         `json:"category"`
	Description string         `json:"description,omitempty"`
	SubjectID   string         `json:"subjectId,omitempty"`
	ReporterID  string         `json:"reporterId,omitempty"`
	Status      string         `json:"status,omitempty"`
	Evidence    map[string]any `json:"evidence,omitempty"`
	CreatedAt   string         `json:"createdAt,omitempty"`
}

// AbuseTriage is the auto-triage verdict returned alongside a created report.
// Severity is the computed risk tier; DedupKey collapses duplicate reports of
// the same subject+category; AutoEscalate routes high-risk categories to a
// human queue; FeedToDetector signals the report should train the firewall;
// Reasons explains the verdict.
type AbuseTriage struct {
	Severity       string   `json:"severity"`
	Category       string   `json:"category"`
	DedupKey       string   `json:"dedupKey"`
	AutoEscalate   bool     `json:"autoEscalate"`
	FeedToDetector bool     `json:"feedToDetector"`
	Reasons        []string `json:"reasons,omitempty"`
}

// ReportAbuseRequest is the POST /abuse-reports body. Category is required and
// must be one of: csam, violence, self_harm, harassment, hate, fraud, privacy,
// spam, other. The rest are optional context.
type ReportAbuseRequest struct {
	ProjectID   string         `json:"projectId"`
	Category    string         `json:"category"`
	Description string         `json:"description,omitempty"`
	SubjectID   string         `json:"subjectId,omitempty"`
	ReporterID  string         `json:"reporterId,omitempty"`
	Evidence    map[string]any `json:"evidence,omitempty"`
}

// ReportAbuseResponse is the 201 payload: the stored report plus its triage.
type ReportAbuseResponse struct {
	Report AbuseReport `json:"report"`
	Triage AbuseTriage `json:"triage"`
}

// listAbuseReportsResponse is the GET /abuse-reports envelope's data object.
type listAbuseReportsResponse struct {
	Reports []AbuseReport `json:"reports"`
}

// ReportAbuse files a trust-and-safety report and returns the stored row plus
// its auto-triage verdict. POST /api/v1/abuse-reports returns 201.
func (c *Client) ReportAbuse(ctx context.Context, req *ReportAbuseRequest) (*ReportAbuseResponse, error) {
	if req == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ReportAbuse: req is required"}
	}
	if req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ReportAbuse: ProjectID is required"}
	}
	if req.Category == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ReportAbuse: Category is required"}
	}
	var result ReportAbuseResponse
	if err := c.doRequest(ctx, http.MethodPost, "/abuse-reports", req, &result); err != nil {
		return nil, fmt.Errorf("ReportAbuse: %w", err)
	}
	// An absent triage read as "no severity, do not escalate, do not feed the
	// detector" — a CSAM or self-harm report dropping silently out of the human
	// review queue.
	if result.Triage.Severity == "" {
		return nil, indeterminateVerdict("ReportAbuse", "triage.severity", "POST /abuse-reports")
	}
	// Bound to the REQUEST: both escalation flags and the dedup key are derived
	// from the category and subject THIS caller filed.
	if reason := result.Triage.inconsistency(req.Category, req.SubjectID); reason != "" {
		return nil, uninterpretableVerdict("ReportAbuse", "POST /abuse-reports", reason)
	}
	return &result, nil
}

// ListAbuseReports returns abuse reports for a project, optionally filtered by
// status. GET /api/v1/abuse-reports?projectId=...&status=... — projectId is
// required; pass status="" for all statuses (otherwise one of open, reviewing,
// actioned, dismissed).
func (c *Client) ListAbuseReports(ctx context.Context, projectID, status string) ([]AbuseReport, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ListAbuseReports: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	if status != "" {
		q.Set("status", status)
	}
	var result listAbuseReportsResponse
	if err := c.doRequest(ctx, http.MethodGet, "/abuse-reports?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListAbuseReports: %w", err)
	}
	return result.Reports, nil
}

// ─── Agent deployments (publish a workflow as a chat widget) ───────────────
//
// Publishes a built agent workflow to a channel (an embeddable web widget,
// Slack, WhatsApp, or a raw API endpoint) and manages the deployment lifecycle.
// Create/list hang off /api/v1/workflows/{workflowId}/deploy; update/delete
// address the deployment directly at /api/v1/deployments/{id}.

// AgentDeployment is a published workflow endpoint. PublicID is the
// caller-shareable handle used to embed/address the widget; the remaining
// fields mirror the deployment row.
type AgentDeployment struct {
	ID             string   `json:"id"`
	PublicID       string   `json:"public_id"`
	WorkflowID     string   `json:"workflow_id,omitempty"`
	ProjectID      string   `json:"project_id,omitempty"`
	Channel        string   `json:"channel"`
	Status         string   `json:"status,omitempty"`
	Greeting       string   `json:"greeting,omitempty"`
	AllowedOrigins []string `json:"allowed_origins,omitempty"`
	CreatedAt      string   `json:"created_at,omitempty"`
	UpdatedAt      string   `json:"updated_at,omitempty"`
}

// listAgentDeploymentsResponse is the GET deploy envelope's data object.
type listAgentDeploymentsResponse struct {
	Deployments []AgentDeployment `json:"deployments"`
}

// DeployAgentRequest is the POST /workflows/{workflowId}/deploy body. Channel
// is required (one of web, slack, whatsapp, api). AllowedOrigins scopes the
// embeddable web widget's CORS; Greeting is the widget's opening message.
type DeployAgentRequest struct {
	ProjectID      string   `json:"projectId"`
	Channel        string   `json:"channel"`
	AllowedOrigins []string `json:"allowedOrigins,omitempty"`
	Greeting       string   `json:"greeting,omitempty"`
}

// UpdateAgentDeploymentRequest is the PATCH /deployments/{id} body. All fields
// beyond ProjectID are optional; nil/empty fields are left unchanged. Status
// toggles the deployment between "active" and "paused".
type UpdateAgentDeploymentRequest struct {
	ProjectID      string    `json:"projectId"`
	Status         string    `json:"status,omitempty"`
	Greeting       *string   `json:"greeting,omitempty"`
	AllowedOrigins *[]string `json:"allowedOrigins,omitempty"`
}

// DeployAgent publishes a workflow to a channel and returns the new deployment
// (including its public_id). POST /api/v1/workflows/{workflowId}/deploy → 201.
func (c *Client) DeployAgent(ctx context.Context, workflowID string, req *DeployAgentRequest) (*AgentDeployment, error) {
	if req == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DeployAgent: req is required"}
	}
	if req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DeployAgent: ProjectID is required"}
	}
	if req.Channel == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DeployAgent: Channel is required"}
	}
	var result AgentDeployment
	path := fmt.Sprintf("/workflows/%s/deploy", url.PathEscape(workflowID))
	if err := c.doRequest(ctx, http.MethodPost, path, req, &result); err != nil {
		return nil, fmt.Errorf("DeployAgent: %w", err)
	}
	return &result, nil
}

// ListAgentDeployments returns the deployments for a workflow.
// GET /api/v1/workflows/{workflowId}/deploy?projectId=... — projectId required.
func (c *Client) ListAgentDeployments(ctx context.Context, workflowID, projectID string) ([]AgentDeployment, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ListAgentDeployments: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	var result listAgentDeploymentsResponse
	path := fmt.Sprintf("/workflows/%s/deploy?%s", url.PathEscape(workflowID), q.Encode())
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("ListAgentDeployments: %w", err)
	}
	return result.Deployments, nil
}

// UpdateAgentDeployment patches a deployment (pause/resume, greeting, origins).
// PATCH /api/v1/deployments/{id} returns the updated deployment.
func (c *Client) UpdateAgentDeployment(ctx context.Context, deploymentID string, req *UpdateAgentDeploymentRequest) (*AgentDeployment, error) {
	if req == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "UpdateAgentDeployment: req is required"}
	}
	if req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "UpdateAgentDeployment: ProjectID is required"}
	}
	var result AgentDeployment
	path := fmt.Sprintf("/deployments/%s", url.PathEscape(deploymentID))
	if err := c.doRequest(ctx, http.MethodPatch, path, req, &result); err != nil {
		return nil, fmt.Errorf("UpdateAgentDeployment: %w", err)
	}
	return &result, nil
}

// DeleteAgentDeployment unpublishes a deployment.
// DELETE /api/v1/deployments/{id}?projectId=... — projectId required.
func (c *Client) DeleteAgentDeployment(ctx context.Context, deploymentID, projectID string) error {
	if projectID == "" {
		return &EvalGuardError{Code: ErrCodeValidation, Message: "DeleteAgentDeployment: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	path := fmt.Sprintf("/deployments/%s?%s", url.PathEscape(deploymentID), q.Encode())
	if err := c.doRequest(ctx, http.MethodDelete, path, nil, nil); err != nil {
		return fmt.Errorf("DeleteAgentDeployment: %w", err)
	}
	return nil
}

// --- Agent memory (two-tier: long-term semantic recall) ---

// MemoryHit is one long-term semantic-recall result. Score is nil when listing
// recent facts without a query.
type MemoryHit struct {
	ID        string   `json:"id,omitempty"`
	Content   string   `json:"content"`
	Score     *float64 `json:"score"`
	CreatedAt string   `json:"createdAt,omitempty"`
}

// MemoryTurn is a conversation turn fed to LLM fact extraction.
type MemoryTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// RememberMemoryResult reports which facts were stored vs. skipped as duplicates.
type RememberMemoryResult struct {
	Written []string `json:"written"`
	Skipped []string `json:"skipped"`
}

// RecallMemoryResult is the recall response's data object.
type RecallMemoryResult struct {
	Semantic []MemoryHit `json:"semantic"`
}

type rememberMemoryBody struct {
	ProjectID  string       `json:"projectId"`
	SessionKey string       `json:"sessionKey"`
	Facts      []string     `json:"facts,omitempty"`
	Turns      []MemoryTurn `json:"turns,omitempty"`
	AgentID    string       `json:"agentId,omitempty"`
}

// RememberMemory stores durable facts (or a conversation to extract facts from)
// for a session. POST /api/v1/agent-memory.
func (c *Client) RememberMemory(ctx context.Context, projectID, sessionKey string, facts []string, turns []MemoryTurn, agentID string) (*RememberMemoryResult, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RememberMemory: projectID is required"}
	}
	if sessionKey == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RememberMemory: sessionKey is required"}
	}
	if len(facts) == 0 && len(turns) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RememberMemory: provide facts or turns"}
	}
	var result RememberMemoryResult
	body := rememberMemoryBody{ProjectID: projectID, SessionKey: sessionKey, Facts: facts, Turns: turns, AgentID: agentID}
	if err := c.doRequest(ctx, http.MethodPost, "/agent-memory", body, &result); err != nil {
		return nil, fmt.Errorf("RememberMemory: %w", err)
	}
	return &result, nil
}

// RecallMemory recalls a session's long-term memory by semantic similarity to
// query (empty query lists recent facts). GET /api/v1/agent-memory.
func (c *Client) RecallMemory(ctx context.Context, projectID, sessionKey, query string, limit int) (*RecallMemoryResult, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RecallMemory: projectID is required"}
	}
	if sessionKey == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RecallMemory: sessionKey is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("sessionKey", sessionKey)
	if query != "" {
		q.Set("query", query)
	}
	if limit > 0 {
		q.Set("limit", fmt.Sprintf("%d", limit))
	}
	var result RecallMemoryResult
	if err := c.doRequest(ctx, http.MethodGet, "/agent-memory?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("RecallMemory: %w", err)
	}
	return &result, nil
}

// ForgetMemory forgets a session's long-term memory, returning the number of
// rows removed. DELETE /api/v1/agent-memory.
func (c *Client) ForgetMemory(ctx context.Context, projectID, sessionKey string) (int, error) {
	if projectID == "" {
		return 0, &EvalGuardError{Code: ErrCodeValidation, Message: "ForgetMemory: projectID is required"}
	}
	if sessionKey == "" {
		return 0, &EvalGuardError{Code: ErrCodeValidation, Message: "ForgetMemory: sessionKey is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("sessionKey", sessionKey)
	var result struct {
		Forgotten int `json:"forgotten"`
	}
	if err := c.doRequest(ctx, http.MethodDelete, "/agent-memory?"+q.Encode(), nil, &result); err != nil {
		return 0, fmt.Errorf("ForgetMemory: %w", err)
	}
	return result.Forgotten, nil
}

// --- Agent-memory governance policy (admin-managed durable-write governance) ---
//
// Mirrors the provider-keys / models-registry CRUD pattern above: an org-scoped
// (optionally project-scoped) resource read/upserted/deleted via orgId +
// optional projectId query params. Admin role required on every verb; the
// server audits each write. See GET/PUT/DELETE /api/v1/agent-memory/governance.

// MemoryGovernanceMode is a policy's enforcement posture:
//   - "off"     — allow everything (governance inert).
//   - "monitor" — record would-be verdicts, but never block a write.
//   - "enforce" — verdicts act (block / require approval).
type MemoryGovernanceMode string

const (
	MemoryGovernanceOff     MemoryGovernanceMode = "off"
	MemoryGovernanceMonitor MemoryGovernanceMode = "monitor"
	MemoryGovernanceEnforce MemoryGovernanceMode = "enforce"
)

// MemoryGovernanceThresholds tunes the numeric governance signals.
type MemoryGovernanceThresholds struct {
	// PoisonMinConfidence is the minimum poisoning-screen confidence (0..1)
	// required to act on a flagged memory. Nil leaves the server default.
	PoisonMinConfidence *float64 `json:"poisonMinConfidence,omitempty"`
}

// MemoryGovernanceConfig is the tunable knob set stored on a policy (the DB
// `config` JSONB). Every field is optional — an empty config keeps server
// defaults for the omitted knobs.
type MemoryGovernanceConfig struct {
	Thresholds *MemoryGovernanceThresholds `json:"thresholds,omitempty"`
	// RequireApprovalOnRewrite gates consolidate/rewrite writes behind HITL approval.
	RequireApprovalOnRewrite *bool `json:"requireApprovalOnRewrite,omitempty"`
	// RequireProvenance flags any governed memory that lacks a non-empty source.
	RequireProvenance *bool `json:"requireProvenance,omitempty"`
}

// MemoryGovernancePolicy is the admin-managed governance policy for durable
// agent-memory writes in an org (optionally scoped to a project). ProjectID and
// CreatedBy are nil for an org-wide, system-seeded policy respectively.
type MemoryGovernancePolicy struct {
	ID        string                 `json:"id"`
	OrgID     string                 `json:"orgId"`
	ProjectID *string                `json:"projectId"`
	Enabled   bool                   `json:"enabled"`
	Mode      MemoryGovernanceMode   `json:"mode"`
	Config    MemoryGovernanceConfig `json:"config"`
	CreatedBy *string                `json:"createdBy"`
	CreatedAt string                 `json:"createdAt"`
	UpdatedAt string                 `json:"updatedAt"`
}

// SetAgentMemoryGovernanceRequest is the upsert body for the org(+project)
// policy. Only OrgID is required; each omitted field keeps its existing value
// (or the server default on first insert).
type SetAgentMemoryGovernanceRequest struct {
	OrgID     string                  `json:"orgId"`
	ProjectID *string                 `json:"projectId,omitempty"`
	Enabled   *bool                   `json:"enabled,omitempty"`
	Mode      MemoryGovernanceMode    `json:"mode,omitempty"`
	Config    *MemoryGovernanceConfig `json:"config,omitempty"`
}

// GetAgentMemoryGovernance reads the org's memory-governance policy for the
// given project scope (pass nil projectID for the org-wide policy). Returns a
// nil policy (and nil error) when none is configured. Admin role required.
// GET /api/v1/agent-memory/governance.
func (c *Client) GetAgentMemoryGovernance(ctx context.Context, orgID string, projectID *string) (*MemoryGovernancePolicy, error) {
	if orgID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "GetAgentMemoryGovernance: orgID is required"}
	}
	q := url.Values{}
	q.Set("orgId", orgID)
	if projectID != nil {
		q.Set("projectId", *projectID)
	}
	var result struct {
		Policy *MemoryGovernancePolicy `json:"policy"`
	}
	if err := c.doRequest(ctx, http.MethodGet, "/agent-memory/governance?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetAgentMemoryGovernance: %w", err)
	}
	return result.Policy, nil
}

// SetAgentMemoryGovernance upserts the org(+project) memory-governance policy
// and returns the stored row. Admin role required; the write is audited
// server-side. PUT /api/v1/agent-memory/governance.
func (c *Client) SetAgentMemoryGovernance(ctx context.Context, req SetAgentMemoryGovernanceRequest) (*MemoryGovernancePolicy, error) {
	if req.OrgID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "SetAgentMemoryGovernance: OrgID is required"}
	}
	var result struct {
		Policy MemoryGovernancePolicy `json:"policy"`
	}
	if err := c.doRequest(ctx, http.MethodPut, "/agent-memory/governance", req, &result); err != nil {
		return nil, fmt.Errorf("SetAgentMemoryGovernance: %w", err)
	}
	return &result.Policy, nil
}

// DeleteAgentMemoryGovernance removes the org(+project) memory-governance policy
// (reverting to no governance) and reports whether a row was removed. Admin role
// required. DELETE /api/v1/agent-memory/governance.
func (c *Client) DeleteAgentMemoryGovernance(ctx context.Context, orgID string, projectID *string) (bool, error) {
	if orgID == "" {
		return false, &EvalGuardError{Code: ErrCodeValidation, Message: "DeleteAgentMemoryGovernance: orgID is required"}
	}
	q := url.Values{}
	q.Set("orgId", orgID)
	if projectID != nil {
		q.Set("projectId", *projectID)
	}
	var result struct {
		Deleted bool `json:"deleted"`
	}
	if err := c.doRequest(ctx, http.MethodDelete, "/agent-memory/governance?"+q.Encode(), nil, &result); err != nil {
		return false, fmt.Errorf("DeleteAgentMemoryGovernance: %w", err)
	}
	return result.Deleted, nil
}

// --- Gateway guardrail config (per-project inline guardrail management) ---
//
// Mirrors the agent-memory governance CRUD above: an org/project-scoped resource
// listed / upserted / deleted over GET/POST/DELETE /api/v1/gateway/guardrails.
// Admin role required on upsert + delete; the server audits each write and busts
// the gateway loader cache so a change takes effect on the very next proxy request.
//
// Each row enables ONE guardrail "vendor" on the project's gateway hot path:
// either a partner vendor adapter (Lakera / Aporia / Patronus / … — resolved
// server-side from an API key referenced by SecretRef) or a LOCAL preset that
// makes NO external call and needs NO secret. The four local vendors are
// local-firewall, moderated-firewall, and the two Wave-2 agent guardrails
// data-not-instructions + tool-call-circuit-breaker. The route REJECTS a local
// vendor that carries a SecretRef (400 SECRET_REF_NOT_ALLOWED) and a non-local
// vendor that omits one (400 SECRET_REF_REQUIRED), so this client models the
// split explicitly and fails fast — see LocalGuardrailVendors / IsLocalGuardrailVendor.

// GuardrailFlagAction is the action a guardrail takes on flagged content:
//   - "block"  — reject the request/response.
//   - "redact" — strip the flagged span and continue.
//   - "flag"   — allow but record the verdict.
type GuardrailFlagAction string

const (
	GuardrailFlagBlock  GuardrailFlagAction = "block"
	GuardrailFlagRedact GuardrailFlagAction = "redact"
	GuardrailFlagFlag   GuardrailFlagAction = "flag"
)

// LocalGuardrailVendors are the guardrail vendors that run inline with NO
// external call and therefore MUST NOT carry a SecretRef. The two Wave-2 agent
// guardrails (data-not-instructions, tool-call-circuit-breaker) join the two
// content presets (local-firewall, moderated-firewall) here.
var LocalGuardrailVendors = []string{
	"local-firewall",
	"moderated-firewall",
	"data-not-instructions",
	"tool-call-circuit-breaker",
}

// IsLocalGuardrailVendor reports whether vendor is a local (no-secret) guardrail
// preset. Local vendors forbid a SecretRef; every other (partner) vendor requires one.
func IsLocalGuardrailVendor(vendor string) bool {
	for _, v := range LocalGuardrailVendors {
		if v == vendor {
			return true
		}
	}
	return false
}

// GuardrailConfig is one per-project gateway guardrail-config row as returned by
// the API (snake_case DB columns). SecretRef is nil for a local preset and for a
// partner vendor whose key binding was cleared.
type GuardrailConfig struct {
	ID               string              `json:"id"`
	OrgID            string              `json:"org_id"`
	ProjectID        string              `json:"project_id"`
	Vendor           string              `json:"vendor"`
	VendorChain      []string            `json:"vendor_chain,omitempty"`
	FallbackOnErrors []string            `json:"fallback_on_errors,omitempty"`
	Config           map[string]any      `json:"config,omitempty"`
	SecretRef        *string             `json:"secret_ref"`
	OnFlag           GuardrailFlagAction `json:"on_flag"`
	CheckRequest     bool                `json:"check_request"`
	CheckResponse    bool                `json:"check_response"`
	TokenizePii      bool                `json:"tokenize_pii"`
	Enabled          bool                `json:"enabled"`
	Priority         int                 `json:"priority"`
	CreatedAt        string              `json:"created_at"`
	UpdatedAt        string              `json:"updated_at"`
}

// UpsertGuardrailConfigRequest is the upsert body for a gateway guardrail-config
// row (camelCase, matching the POST schema). OrgID, ProjectID and Vendor are
// required; every other field is optional and keeps the server default on first
// insert. SecretRef MUST be set for a partner vendor and MUST be nil for a local
// vendor (see IsLocalGuardrailVendor). Pointer bools/ints distinguish "leave the
// existing value" (nil) from an explicit false/0.
type UpsertGuardrailConfigRequest struct {
	OrgID            string              `json:"orgId"`
	ProjectID        string              `json:"projectId"`
	Vendor           string              `json:"vendor"`
	VendorChain      []string            `json:"vendorChain,omitempty"`
	FallbackOnErrors []string            `json:"fallbackOnErrors,omitempty"`
	Config           map[string]any      `json:"config,omitempty"`
	SecretRef        *string             `json:"secretRef,omitempty"`
	OnFlag           GuardrailFlagAction `json:"onFlag,omitempty"`
	CheckRequest     *bool               `json:"checkRequest,omitempty"`
	CheckResponse    *bool               `json:"checkResponse,omitempty"`
	TokenizePii      *bool               `json:"tokenizePii,omitempty"`
	Enabled          *bool               `json:"enabled,omitempty"`
	Priority         *int                `json:"priority,omitempty"`
}

// ListGuardrailConfigs returns the project's gateway guardrail-config rows,
// ordered by priority (ascending). Returns an empty slice (and nil error) when
// none are configured. Admin auth. GET /api/v1/gateway/guardrails?projectId=….
func (c *Client) ListGuardrailConfigs(ctx context.Context, projectID string) ([]GuardrailConfig, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ListGuardrailConfigs: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	var result []GuardrailConfig
	if err := c.doRequest(ctx, http.MethodGet, "/gateway/guardrails?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("ListGuardrailConfigs: %w", err)
	}
	return result, nil
}

// UpsertGuardrailConfig enables/configures ONE guardrail vendor on the project's
// gateway hot path and returns the stored row. Idempotent on (projectId, vendor):
// re-submitting the same vendor updates in place. Admin role required; the write
// is audited server-side and busts the gateway loader cache.
// POST /api/v1/gateway/guardrails.
//
// The local-vs-vendor SecretRef rule is enforced client-side to fail fast (the
// server 400s the same mismatches): a LOCAL vendor (see IsLocalGuardrailVendor)
// must NOT carry a SecretRef; every partner vendor MUST. A supplied VendorChain
// must lead with the primary Vendor (its (projectId, vendor) upsert key).
func (c *Client) UpsertGuardrailConfig(ctx context.Context, req UpsertGuardrailConfigRequest) (*GuardrailConfig, error) {
	if req.OrgID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "UpsertGuardrailConfig: OrgID is required"}
	}
	if req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "UpsertGuardrailConfig: ProjectID is required"}
	}
	if req.Vendor == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "UpsertGuardrailConfig: Vendor is required"}
	}
	if IsLocalGuardrailVendor(req.Vendor) {
		if req.SecretRef != nil {
			return nil, &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("UpsertGuardrailConfig: local guardrail %q makes no external call and must not carry a SecretRef", req.Vendor)}
		}
	} else if req.SecretRef == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("UpsertGuardrailConfig: partner vendor %q requires a SecretRef pointing at a stored provider key", req.Vendor)}
	}
	if len(req.VendorChain) > 0 && req.VendorChain[0] != req.Vendor {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("UpsertGuardrailConfig: VendorChain[0] (%q) must equal Vendor (%q) — the primary", req.VendorChain[0], req.Vendor)}
	}
	var result GuardrailConfig
	if err := c.doRequest(ctx, http.MethodPost, "/gateway/guardrails", req, &result); err != nil {
		return nil, fmt.Errorf("UpsertGuardrailConfig: %w", err)
	}
	return &result, nil
}

// DeleteGuardrailConfig removes ONE guardrail-config row by id (scoped to the
// project) and returns the deleted row id. Admin role required; the delete is
// audited server-side and busts the gateway loader cache.
// DELETE /api/v1/gateway/guardrails?projectId=…&id=….
func (c *Client) DeleteGuardrailConfig(ctx context.Context, projectID, id string) (string, error) {
	if projectID == "" {
		return "", &EvalGuardError{Code: ErrCodeValidation, Message: "DeleteGuardrailConfig: projectID is required"}
	}
	if id == "" {
		return "", &EvalGuardError{Code: ErrCodeValidation, Message: "DeleteGuardrailConfig: id is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	q.Set("id", id)
	var result struct {
		Deleted string `json:"deleted"`
	}
	if err := c.doRequest(ctx, http.MethodDelete, "/gateway/guardrails?"+q.Encode(), nil, &result); err != nil {
		return "", fmt.Errorf("DeleteGuardrailConfig: %w", err)
	}
	return result.Deleted, nil
}

// --- Voice ML (word-level ASR + deepfake detection via sidecar) ---

// VoiceWord is a single word with its time span (ms relative to audio start).
type VoiceWord struct {
	Word       string   `json:"word"`
	StartMs    int      `json:"startMs"`
	EndMs      int      `json:"endMs"`
	Confidence *float64 `json:"confidence,omitempty"`
}

// VoiceSegment is a coarse transcript span.
type VoiceSegment struct {
	StartMs int    `json:"startMs"`
	EndMs   int    `json:"endMs"`
	Text    string `json:"text"`
}

// TranscriptResult is the word-level ASR result.
type TranscriptResult struct {
	Language   string         `json:"language,omitempty"`
	DurationMs int            `json:"durationMs,omitempty"`
	Text       string         `json:"text"`
	Words      []VoiceWord    `json:"words"`
	Segments   []VoiceSegment `json:"segments,omitempty"`
}

// DeepfakeScore is the synthetic-speech detection result; Probability is P(synthetic) in [0,1].
// DeepfakeScore is the synthetic-speech likelihood for a voice sample.
//
// Same CLASS-1 hazard: `Probability` zero-values to 0.0, which is the MOST
// benign reading on a 0..1 scale, so the natural gate
// `if score.Probability > threshold { reject }` accepted every sample whose
// score never arrived. Presence of `probability` on the wire is tracked so
// ScoreVoiceDeepfake can refuse instead.
type DeepfakeScore struct {
	Probability float64 `json:"probability"`
	Model       string  `json:"model,omitempty"`

	// probabilityPresent records whether the wire carried `probability`.
	probabilityPresent bool
}

// UnmarshalJSON decodes the score and records whether `probability` was
// present. An explicit 0.0 (a real "certainly authentic") and an absent field
// are indistinguishable by value.
func (r *DeepfakeScore) UnmarshalJSON(data []byte) error {
	type alias DeepfakeScore
	var probe struct {
		Probability *float64 `json:"probability"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = DeepfakeScore(decoded)
	r.probabilityPresent = probe.Probability != nil
	return nil
}

// HasVerdict reports whether the backend actually returned a probability.
func (r *DeepfakeScore) HasVerdict() bool { return r != nil && r.probabilityPresent }

type voiceBody struct {
	ProjectID   string `json:"projectId"`
	AudioBase64 string `json:"audioBase64"`
	Language    string `json:"language,omitempty"`
}

// TranscribeVoice transcribes base64-encoded WAV audio with WORD-LEVEL
// timestamps. POST /api/v1/voice/transcribe. Requires the operator-deployed
// voice-ML sidecar (returns an error wrapping HTTP 503 otherwise).
func (c *Client) TranscribeVoice(ctx context.Context, projectID, audioBase64, language string) (*TranscriptResult, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "TranscribeVoice: projectID is required"}
	}
	if audioBase64 == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "TranscribeVoice: audioBase64 is required"}
	}
	var result TranscriptResult
	body := voiceBody{ProjectID: projectID, AudioBase64: audioBase64, Language: language}
	if err := c.doRequest(ctx, http.MethodPost, "/voice/transcribe", body, &result); err != nil {
		return nil, fmt.Errorf("TranscribeVoice: %w", err)
	}
	return &result, nil
}

// ScoreVoiceDeepfake scores base64-encoded WAV audio for synthetic-speech /
// deepfake probability in [0,1]. POST /api/v1/voice/deepfake-score.
func (c *Client) ScoreVoiceDeepfake(ctx context.Context, projectID, audioBase64 string) (*DeepfakeScore, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ScoreVoiceDeepfake: projectID is required"}
	}
	if audioBase64 == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ScoreVoiceDeepfake: audioBase64 is required"}
	}
	var result DeepfakeScore
	body := voiceBody{ProjectID: projectID, AudioBase64: audioBase64}
	if err := c.doRequest(ctx, http.MethodPost, "/voice/deepfake-score", body, &result); err != nil {
		return nil, fmt.Errorf("ScoreVoiceDeepfake: %w", err)
	}
	if !result.HasVerdict() {
		return nil, indeterminateVerdict("ScoreVoiceDeepfake", "probability",
			"POST /voice/deepfake-score")
	}
	return &result, nil
}

// --- Language detection (text → language) ---

// LanguageDetection is the result of text language identification (franc-min).
type LanguageDetection struct {
	ISO6393    string  `json:"iso6393"`
	ISO6391    *string `json:"iso6391"`
	Name       *string `json:"name"`
	Confidence float64 `json:"confidence"`
	Reliable   bool    `json:"reliable"`
}

type detectLanguageBody struct {
	ProjectID string `json:"projectId"`
	Text      string `json:"text"`
	MinLength int    `json:"minLength,omitempty"`
}

// DetectLanguage identifies the language of a text snippet. POST /api/v1/language/detect.
func (c *Client) DetectLanguage(ctx context.Context, projectID, text string) (*LanguageDetection, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DetectLanguage: projectID is required"}
	}
	if text == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DetectLanguage: text is required"}
	}
	var result LanguageDetection
	body := detectLanguageBody{ProjectID: projectID, Text: text}
	if err := c.doRequest(ctx, http.MethodPost, "/language/detect", body, &result); err != nil {
		return nil, fmt.Errorf("DetectLanguage: %w", err)
	}
	return &result, nil
}

// --- RAG (retrieval-augmented generation) ---

// RAGDocument is a single document/chunk to ingest into the RAG pipeline.
type RAGDocument struct {
	ID       string         `json:"id,omitempty"`
	Text     string         `json:"text"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// RAGChunking controls how documents are split before embedding.
type RAGChunking struct {
	Strategy     string `json:"strategy,omitempty"` // "fixed" | "recursive"
	ChunkSize    int    `json:"chunkSize,omitempty"`
	ChunkOverlap int    `json:"chunkOverlap,omitempty"`
}

// IngestRAGRequest is the payload for IngestRAGDocuments.
type IngestRAGRequest struct {
	ProjectID  string        `json:"projectId"`
	Documents  []RAGDocument `json:"documents"`
	Chunking   *RAGChunking  `json:"chunking,omitempty"`
	Embed      bool          `json:"embed,omitempty"`
	EmbedModel string        `json:"embedModel,omitempty"`
}

// IngestRAGDocuments chunks (and optionally embeds) documents through the RAG
// ingest pipeline, which also runs DLP + prompt-injection screening on every
// chunk. POST /api/v1/rag/ingest. Returns the chunks plus dlp/injection reports.
// Embedding uses the tenant's BYOK OpenAI key (projectId-scoped).
func (c *Client) IngestRAGDocuments(ctx context.Context, req *IngestRAGRequest) (map[string]any, error) {
	if req == nil || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "IngestRAGDocuments: projectID is required"}
	}
	if len(req.Documents) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "IngestRAGDocuments: at least one document is required"}
	}
	var verdict RAGIngestResult
	result, err := c.postGuardedMap(ctx, "/rag/ingest", req, &verdict)
	if err != nil {
		return nil, fmt.Errorf("IngestRAGDocuments: %w", err)
	}
	// This path runs the SAME DLP + prompt-injection screening ScanRAGInjection
	// was hardened for; an absent report is screening that did not happen, not
	// documents that came back clean.
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("IngestRAGDocuments", f, "POST /rag/ingest")
	}
	if reason := verdict.inconsistency(len(req.Documents)); reason != "" {
		return nil, uninterpretableVerdict("IngestRAGDocuments", "POST /rag/ingest", reason)
	}
	return result, nil
}

// RAGInjectionDocument is one document/chunk to vet for embedded prompt injection.
type RAGInjectionDocument struct {
	Text   string `json:"text"`
	Source string `json:"source,omitempty"`
}

// RAGInjectionScanResult is the outcome of ScanRAGInjection.
//
// Same CLASS-1 hazard as the firewall response: `Clean` zero-values to false
// and `PoisonedIndices` to nil. A caller that keeps every document NOT named in
// PoisonedIndices — the natural filtering idiom — therefore forwarded the whole
// retrieved set to the model when the scan result never arrived. Presence of
// `clean` on the wire is tracked so ScanRAGInjection can refuse.
type RAGInjectionScanResult struct {
	Scanned         int              `json:"scanned"`
	Clean           bool             `json:"clean"`
	PoisonedCount   int              `json:"poisonedCount"`
	PoisonedIndices []int            `json:"poisonedIndices"`
	Violations      []map[string]any `json:"violations"`

	// cleanPresent records whether the wire carried a boolean `clean`.
	cleanPresent bool
}

// UnmarshalJSON decodes the result and records whether `clean` was present.
// See FirewallCheckResponse.UnmarshalJSON for why the alias indirection is
// required and why presence cannot be inferred from the value.
func (r *RAGInjectionScanResult) UnmarshalJSON(data []byte) error {
	type alias RAGInjectionScanResult
	var probe struct {
		Clean *bool `json:"clean"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = RAGInjectionScanResult(decoded)
	r.cleanPresent = probe.Clean != nil
	return nil
}

// ragSeverityRank is the CLOSED severity ladder the chunk scanner ranks against
// (packages/core/src/gateway/rag-injector.ts `SEV_RANK`), and `minSeverity` on
// the route is the same four-value enum. A severity outside it cannot be ranked,
// so a violation carrying one cannot be compared to the acting threshold — that
// is drift, and the scan is refused rather than silently ignored.
var ragSeverityRank = map[string]int{"low": 0, "medium": 1, "high": 2, "critical": 3}

// ragDefaultMinSeverity mirrors the route's server-side default
// (`scanChunksForInjection(docs, { minSeverity: b.minSeverity })` →
// `opts.minSeverity ?? "high"`), applied when the caller passes "".
const ragDefaultMinSeverity = "high"

// ragScanBinding ties a decoded result back to the REQUEST that produced it.
// Nothing on the wire can restate these: docCount is what the CALLER submitted
// and minSeverity is what the CALLER asked for, so a rewritten body cannot move
// the goalposts by editing a field.
type ragScanBinding struct {
	// docCount is the number of documents submitted; -1 when unknown (decoding
	// a stored report, where there is no request to bind to).
	docCount int
	// minSeverity is the threshold the caller requested ("" → the route
	// default). When it is not a rankable severity the threshold is treated as
	// UNKNOWN and the weaker, threshold-free feasibility rule is used instead.
	minSeverity string
}

// HasVerdict reports whether the scan returned a verdict the caller can ACT ON.
//
// That is presence AND self-consistency, because for this result type the two
// are the same safety property. `Clean=false` with an empty `PoisonedIndices`
// LOOKS fail-closed — the scan is reporting poison — while the consequence is
// fail-open: the filtering idiom this SDK documents ("keep every document not
// named as poisoned", CHANGELOG 1.5.0) names nothing to drop and forwards the
// ENTIRE retrieved set to the model. A scan that says unsafe but cannot say
// WHICH documents is INDETERMINATE, not a verdict.
func (r *RAGInjectionScanResult) HasVerdict() bool {
	return r != nil && r.cleanPresent && r.inconsistency(ragScanBinding{docCount: -1}) == ""
}

// violationEvidence is the incrimination the report makes about itself: for each
// document index, the HIGHEST severity rank among the violations the body lists
// against it. Absent from the map ⇒ the report names no violation for that
// document at all.
//
// It returns a reason string when a violation cannot be read as evidence.
// `flagged` entries are emitted by the backend with BOTH `chunkIndex` and
// `severity` on every element (rag-injector.ts pushes a fully populated
// ChunkScanViolation), so a violation missing either one is not a violation this
// client can weigh — and an unweighable violation is exactly how a rewritten
// body would hide a poisoned document from the coverage rule below.
func (r *RAGInjectionScanResult) violationEvidence(bound int) (map[int]int, string) {
	evidence := make(map[int]int, len(r.Violations))
	for n, v := range r.Violations {
		if v == nil {
			return nil, fmt.Sprintf("`violations[%d]` is null and names no document", n)
		}
		raw, ok := v["chunkIndex"]
		if !ok {
			return nil, fmt.Sprintf("`violations[%d]` carries no `chunkIndex`, so the violation "+
				"cannot be attributed to a document", n)
		}
		f, ok := raw.(float64)
		if !ok || f != math.Trunc(f) {
			return nil, fmt.Sprintf("`violations[%d].chunkIndex` is %v, not a document index", n, raw)
		}
		idx := int(f)
		if idx < 0 || idx >= bound {
			return nil, fmt.Sprintf("`violations[%d].chunkIndex` is %d but only %d document(s) were "+
				"scanned — the violation does not point at a document the caller can drop", n, idx, bound)
		}
		sev, _ := v["severity"].(string)
		rank, ok := ragSeverityRank[sev]
		if !ok {
			return nil, fmt.Sprintf("`violations[%d].severity` is %s, outside "+
				"critical/high/medium/low, so it cannot be compared to `minSeverity`", n, quoteVerdict(sev))
		}
		if cur, seen := evidence[idx]; !seen || rank > cur {
			evidence[idx] = rank
		}
	}
	return evidence, ""
}

// inconsistency describes why the result contradicts itself or the request that
// produced it, or "" when it is coherent.
//
// AUDIT 2026-08-03 ROUND 2 — this was a CARDINALITY check standing in for an
// IDENTITY check. `poisonedCount == len(poisonedIndices)` is satisfied by
// `{"poisonedCount":2,"poisonedIndices":[0,0]}`, which identifies ONE document
// while claiming two, and by `[1,4]` when the poisoned documents were 1 and 3 —
// same count, same range, DIFFERENT SET. Either way the documented filter
// forwards an attack document. A count that agrees with a length says nothing
// about WHICH documents were identified, so the check now validates the SET:
// canonical (unique, ascending, in range), bound to the submitted input, and
// equal to the set the report's OWN violations incriminate.
//
// Every rule mirrors an invariant the backend GUARANTEES, so no response a
// healthy server can produce is rejected (that is the over-block control).
// packages/core/src/gateway/rag-injector.ts builds `poisoned` as a Set and
// returns `[...poisoned].sort((a,b) => a-b)` over chunk indices of the submitted
// set, adding i for every violation whose rank ≥ minSeverity; the route
// (apps/web/src/app/api/v1/security/rag-injection-scan/route.ts) sets
// `scanned: documents.length`, `poisonedCount: poisonedIndices.length` and
// `violations: flagged`. Therefore:
//
//	scanned == len(documents)                      (coverage)
//	indices unique + strictly ascending + < scanned (canonical set)
//	clean ⟺ len(indices) == 0                      (agreement)
//	poisonedCount == len(indices)                  (cardinality, still)
//	{i : maxViolationRank(i) ≥ minSeverity} == set(indices)   (identity)
//
// A body violating any of them is drift, a proxy envelope, a replayed scan of a
// different input, or a rewritten payload — never a real verdict.
func (r *RAGInjectionScanResult) inconsistency(b ragScanBinding) string {
	if r == nil {
		return "nil result"
	}
	if !r.cleanPresent {
		// Absence is reported separately, with its own message.
		return ""
	}

	// ── COVERAGE ────────────────────────────────────────────────────────────
	// `scanned` is the backend restating how many documents it actually looked
	// at. Checked FIRST and unconditionally: a verdict about 3 of 5 documents
	// is not a verdict about the retrieved set, however self-consistent the
	// rest of the body is. This also catches an absent `scanned` (0 ≠ n) and a
	// genuine scan of a truncated input replayed against the full request.
	if b.docCount >= 0 {
		if r.Scanned != b.docCount {
			return fmt.Sprintf("`scanned` is %d but %d document(s) were submitted — "+
				"%d document(s) were never examined and the scan says nothing about them",
				r.Scanned, b.docCount, b.docCount-r.Scanned)
		}
	} else if r.Scanned < 1 {
		return "`scanned` is 0 — the report covers no document"
	}

	// ── CANONICAL SET ───────────────────────────────────────────────────────
	// A Set serialized through sort() is unique and strictly ascending. A
	// repeated index inflates the count without naming a second document; a
	// descending pair is a rewritten array.
	named := make(map[int]bool, len(r.PoisonedIndices))
	prev := -1
	for _, i := range r.PoisonedIndices {
		if i < 0 || i >= r.Scanned {
			return fmt.Sprintf("`poisonedIndices` names document %d but only %d were scanned — "+
				"the index does not identify a document the caller can drop", i, r.Scanned)
		}
		if named[i] {
			return fmt.Sprintf("`poisonedIndices` %v repeats document %d — it names %d distinct "+
				"document(s) while claiming %d, so %d poisoned document(s) are unidentified and "+
				"WOULD BE FORWARDED", r.PoisonedIndices, i, len(named), r.PoisonedCount,
				r.PoisonedCount-len(named))
		}
		if i <= prev {
			return fmt.Sprintf("`poisonedIndices` %v is not ascending — the backend serializes a "+
				"sorted Set, so this array was rewritten in transit", r.PoisonedIndices)
		}
		named[i] = true
		prev = i
	}

	// ── AGREEMENT ───────────────────────────────────────────────────────────
	if !r.Clean && len(named) == 0 {
		return fmt.Sprintf("`clean` is false (poison REPORTED) but `poisonedIndices` names no document "+
			"(poisonedCount=%d) — the scan cannot say WHICH documents to drop", r.PoisonedCount)
	}
	if r.Clean && len(named) > 0 {
		return fmt.Sprintf("`clean` is true but `poisonedIndices` names %d document(s) %v",
			len(named), r.PoisonedIndices)
	}
	if r.PoisonedCount != len(r.PoisonedIndices) {
		return fmt.Sprintf("`poisonedCount` is %d but `poisonedIndices` names %d document(s) — "+
			"%d poisoned document(s) are unattributed", r.PoisonedCount, len(r.PoisonedIndices),
			r.PoisonedCount-len(r.PoisonedIndices))
	}

	// ── IDENTITY ────────────────────────────────────────────────────────────
	// The drop-list must be exactly the set of documents the report's own
	// violations incriminate at the acting threshold. This is the only check
	// that ties the indices to WHICH documents are poisoned rather than HOW
	// MANY, and it is evaluated unconditionally: deleting `violations` does not
	// disable it, it makes a non-clean result unevidenced and therefore refused.
	evidence, why := r.violationEvidence(r.Scanned)
	if why != "" {
		return why
	}
	if t, ok := ragSeverityRank[b.minSeverity]; ok || b.minSeverity == "" {
		if b.minSeverity == "" {
			t = ragSeverityRank[ragDefaultMinSeverity]
		}
		return ragIdentityMismatch(named, evidence, t, r.PoisonedIndices)
	}
	// The threshold is unknown (a stored report, or a minSeverity this client
	// cannot rank). Fall back to the threshold-FREE form of the same rule: some
	// threshold in the enum must reproduce this exact drop-list from this exact
	// evidence. It still rejects a named document with no violation at all and
	// an unnamed document carrying a critical one.
	return ragIdentityFeasible(named, evidence, r.PoisonedIndices)
}

// ragIdentityMismatch compares the claimed drop-list against the documents the
// violations incriminate at threshold t.
func ragIdentityMismatch(named map[int]bool, evidence map[int]int, t int, indices []int) string {
	for i := range named {
		rank, ok := evidence[i]
		if !ok {
			return fmt.Sprintf("`poisonedIndices` %v names document %d, but `violations` records "+
				"nothing against it — the drop-list does not match the evidence in the same body", indices, i)
		}
		if rank < t {
			return fmt.Sprintf("`poisonedIndices` names document %d whose worst violation is below "+
				"the requested minSeverity — the drop-list does not match the evidence", i)
		}
	}
	for i, rank := range evidence {
		if rank >= t && !named[i] {
			return fmt.Sprintf("`violations` records a %s-or-worse payload in document %d but "+
				"`poisonedIndices` %v does not name it — that document WOULD BE FORWARDED to the model",
				ragSeverityName(t), i, indices)
		}
	}
	return ""
}

// ragIdentityFeasible is the threshold-free form: it asks whether ANY severity
// in the enum could have produced this drop-list from this evidence.
func ragIdentityFeasible(named map[int]bool, evidence map[int]int, indices []int) string {
	lo := len(ragSeverityRank) - 1 // highest rank a threshold may take ("critical")
	for i := range named {
		rank, ok := evidence[i]
		if !ok {
			return fmt.Sprintf("`poisonedIndices` %v names document %d, but `violations` records "+
				"nothing against it — the drop-list does not match the evidence in the same body", indices, i)
		}
		if rank < lo {
			lo = rank
		}
	}
	hi := -1
	for i, rank := range evidence {
		if !named[i] && rank > hi {
			hi = rank
		}
	}
	if hi >= lo {
		return fmt.Sprintf("no minSeverity explains `poisonedIndices` %v: an unnamed document carries "+
			"a %s payload while a named one carries only %s — that document WOULD BE FORWARDED",
			indices, ragSeverityName(hi), ragSeverityName(lo))
	}
	return ""
}

// ragSeverityName renders a rank back to its severity label for an operator
// message.
func ragSeverityName(rank int) string {
	for name, r := range ragSeverityRank {
		if r == rank {
			return name
		}
	}
	return fmt.Sprintf("rank-%d", rank)
}

type scanRAGInjectionBody struct {
	ProjectID   string                 `json:"projectId,omitempty"`
	Documents   []RAGInjectionDocument `json:"documents"`
	MinSeverity string                 `json:"minSeverity,omitempty"`
}

// ScanRAGInjection screens retrieved documents/chunks for embedded prompt
// injection ("poisoned" context) before they reach the model. minSeverity
// defaults to "high" server-side. POST /api/v1/security/rag-injection-scan.
func (c *Client) ScanRAGInjection(ctx context.Context, projectID string, documents []RAGInjectionDocument, minSeverity string) (*RAGInjectionScanResult, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ScanRAGInjection: projectID is required"}
	}
	if len(documents) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ScanRAGInjection: at least one document is required"}
	}
	var result RAGInjectionScanResult
	body := scanRAGInjectionBody{ProjectID: projectID, Documents: documents, MinSeverity: minSeverity}
	if err := c.doRequest(ctx, http.MethodPost, "/security/rag-injection-scan", body, &result); err != nil {
		return nil, fmt.Errorf("ScanRAGInjection: %w", err)
	}
	if !result.cleanPresent {
		return nil, indeterminateVerdict("ScanRAGInjection", "clean",
			"POST /security/rag-injection-scan")
	}
	// The result is validated against the REQUEST — how many documents were
	// submitted and at what threshold — because only the caller side knows
	// those. A response can restate neither, so it cannot move the goalposts:
	// an index the caller cannot map back to a submitted document is as useless
	// as no index at all, and a scan that covered fewer documents than were
	// submitted is not a verdict about the retrieved set.
	if reason := result.inconsistency(ragScanBinding{docCount: len(documents), minSeverity: minSeverity}); reason != "" {
		return nil, uninterpretableVerdict("ScanRAGInjection",
			"POST /security/rag-injection-scan", reason)
	}
	return &result, nil
}

// --- Multimodal moderation (image / video / deepfake) ---
//
// AUDIT 2026-08-06 (fail-open sweep, CLASS 1 — third pass). ModerateImage,
// ModerateVideo and DetectMediaDeepfake carried the SAME defect 1.5.0 closed on
// /firewall/check, reached through a different return type. They hand back
// map[string]any and the caller's gate is `res["flagged"].(bool)` /
// `res["synthetic"].(bool)`; a type assertion on an ABSENT key yields the zero
// value, and the one-value form does not panic on a missing key the way it does
// on a wrong type. So `{}`, an empty body, `null`, `{"data":null}`, an envelope
// with no verdict, and any unrelated HTTP 200 all read as "clean image,
// authentic media" with err == nil.
//
// ModerateImage's doc comment used to end "Fails closed." That was a claim about
// the SERVER engine (moderateImage() in packages/core/src/image/moderation.ts
// returns flagged:true when the vision backend throws). The Go CLIENT did the
// opposite — it degraded to a zero-valued map and reported success. The comments
// below now describe the client's own behaviour.
//
// The map[string]any return type is kept ON PURPOSE. 1.5.0 changed BEHAVIOUR (an
// unreadable verdict becomes an error) and only ADDED public API; it broke no
// signature, which is what let it ship as a MINOR. Swapping these three to typed
// returns would break every existing caller's compile. So the typed,
// presence-tracking structs below are decoded ALONGSIDE the map, they decide the
// refusal, and the map is handed back untouched when — and only when — the
// verdict is real. HasVerdict is exported on each for the same reason it is on
// FirewallCheckResponse: a caller decoding a stored or proxied body itself needs
// the identical check.

// Server-side defaults from packages/core/src/image (`opts.threshold ?? 0.7`,
// `?? 0.5`, `?? 16`, `?? 1`). The request structs use `omitempty`, so a zero
// value never reaches the wire and the server applies these — mirroring them is
// what lets the client re-derive the verdict from the same inputs the engine had.
const (
	defaultVisionModerationThreshold = 0.7
	defaultMediaDeepfakeThreshold    = 0.5
	defaultMediaMaxFrames            = 16
	defaultMediaSampleEveryN         = 1
)

// mediaFloatTolerance absorbs the last-bit difference an aggregate can pick up
// crossing the wire. Selections (max) and comparisons are checked EXACTLY;
// only the running mean, which accumulates rounding, is compared with slack.
const mediaFloatTolerance = 1e-9

// mediaBinding carries what only the CALLER knows: the request this response is
// supposed to be about. A response can restate none of it, so it cannot move its
// own goalposts — the same reason AuditMcpServer binds to len(tools).
//
// The zero-ish sentinels mean "unknown", for a caller decoding a stored body
// through HasVerdict() with no request in hand.
type mediaBinding struct {
	threshold    float64 // < 0: unknown
	frameCount   int     // < 0: unknown
	maxFrames    int
	sampleEveryN int
	kind         string // "": unknown
}

// unknownMediaBinding drops every request-side check and keeps the ones a body
// can be judged against on its own terms.
var unknownMediaBinding = mediaBinding{threshold: -1, frameCount: -1}

// expectedSampledFrames mirrors moderateVideoFrames() / detectVideoDeepfake()
// exactly: `frames.filter((_, i) => i % sampleEveryN === 0).slice(0, maxFrames)`,
// i.e. min(maxFrames, ceil(total / sampleEveryN)).
func expectedSampledFrames(total, sampleEveryN, maxFrames int) int {
	if sampleEveryN < 1 {
		sampleEveryN = 1
	}
	if maxFrames < 1 {
		maxFrames = 1
	}
	n := (total + sampleEveryN - 1) / sampleEveryN
	if n > maxFrames {
		n = maxFrames
	}
	return n
}

// sameStringSet reports whether got and want hold the same distinct values.
// Order is not part of the contract (the engine builds the union from a Set).
func sameStringSet(got, want []string) bool {
	g := map[string]bool{}
	for _, s := range got {
		g[s] = true
	}
	w := map[string]bool{}
	for _, s := range want {
		w[s] = true
	}
	if len(g) != len(w) {
		return false
	}
	for s := range w {
		if !g[s] {
			return false
		}
	}
	return true
}

// ImageModerationResult is the typed view of a POST /moderation/image body
// (VisionModerationResult in packages/core/src/image/moderation.ts).
//
// `Flagged` zero-values to false and `Score` to 0.0 — on a 0..1 harm scale that
// is the most benign reading available, so a 200 that is not a moderation result
// read as "clean image, zero harm". Presence of each is tracked separately
// because an explicit `flagged:false` (a real allow) and an absent `flagged` are
// INDISTINGUISHABLE by value.
type ImageModerationResult struct {
	Flagged        bool               `json:"flagged"`
	Score          float64            `json:"score"`
	Categories     []string           `json:"categories"`
	CategoryScores map[string]float64 `json:"categoryScores,omitempty"`
	Provider       string             `json:"provider,omitempty"`
	LatencyMs      float64            `json:"latencyMs,omitempty"`

	flaggedPresent bool
	scorePresent   bool
}

// UnmarshalJSON decodes the result and, separately, records which decision
// fields the wire actually carried. The `alias` indirection is what keeps this
// from recursing — a defined type with the same underlying struct has an empty
// method set, so every other field is populated by the default decoder exactly
// as before.
func (r *ImageModerationResult) UnmarshalJSON(data []byte) error {
	type alias ImageModerationResult
	var probe struct {
		Flagged *bool    `json:"flagged"`
		Score   *float64 `json:"score"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = ImageModerationResult(decoded)
	r.flaggedPresent = probe.Flagged != nil
	r.scorePresent = probe.Score != nil
	return nil
}

// missingVerdictField names the decision field the body did not carry, or "".
func (r *ImageModerationResult) missingVerdictField() string {
	switch {
	case r == nil, !r.flaggedPresent:
		return "flagged"
	case !r.scorePresent:
		return "score"
	}
	return ""
}

// HasVerdict reports whether this body carried a moderation verdict this client
// can ACT ON. POST /moderation/image always emits both `flagged` and `score`, so
// false here means the body did not come from the moderation engine.
func (r *ImageModerationResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(unknownMediaBinding) == ""
}

// inconsistency describes why the result contradicts itself or the request that
// produced it, or "" when it is coherent.
func (r *ImageModerationResult) inconsistency(b mediaBinding) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	// clamp01() in moderation.ts guarantees the range; anything else did not
	// come through the engine, and a threshold comparison against it is
	// meaningless.
	if r.Score < 0 || r.Score > 1 {
		return fmt.Sprintf("`score` is %v, outside the 0..1 range the engine clamps to", r.Score)
	}
	// DERIVATION — moderateImage() computes
	// `flagged = backend.flagged === true || score >= threshold`, so a score at
	// or above the threshold THIS caller asked for cannot coexist with
	// `flagged:false`. The converse is legitimate: the backend may flag on its
	// own below the threshold, so a low-score flag is not a contradiction.
	if b.threshold >= 0 && !r.Flagged && r.Score >= b.threshold {
		return fmt.Sprintf("`flagged` is false but `score` is %v at threshold %v — the engine derives "+
			"`flagged` from `score >= threshold`, so these cannot both be true", r.Score, b.threshold)
	}
	return ""
}

// VideoModerationFrame is one moderated frame inside a VideoModerationResult.
type VideoModerationFrame struct {
	Index       int      `json:"index"`
	TimestampMs *float64 `json:"timestampMs,omitempty"`
	Flagged     bool     `json:"flagged"`
	Score       float64  `json:"score"`
	Categories  []string `json:"categories"`
}

// VideoModerationResult is the typed view of a POST /moderation/video body
// (VideoModerationResult in packages/core/src/image/moderation.ts).
//
// Same zero-value hazard as the image path, plus a coverage one: `FramesTotal`
// and `FramesEvaluated` both zero-value to 0, so "clean clip" and "no frame was
// ever moderated" were the same answer.
type VideoModerationResult struct {
	Flagged           bool                   `json:"flagged"`
	Score             float64                `json:"score"`
	Categories        []string               `json:"categories"`
	FirstFlaggedFrame *int                   `json:"firstFlaggedFrame,omitempty"`
	FramesTotal       int                    `json:"framesTotal"`
	FramesEvaluated   int                    `json:"framesEvaluated"`
	Frames            []VideoModerationFrame `json:"frames"`
	Provider          string                 `json:"provider,omitempty"`
	LatencyMs         float64                `json:"latencyMs,omitempty"`

	flaggedPresent bool
	scorePresent   bool
	framesPresent  bool
}

// UnmarshalJSON decodes the clip verdict and records which decision fields the
// wire carried. `frames` is tracked too: it is the EVIDENCE the clip verdict is
// aggregated from, and requiring it is what stops a rewritten body from deleting
// the evidence to escape the derivation checks below.
func (r *VideoModerationResult) UnmarshalJSON(data []byte) error {
	type alias VideoModerationResult
	var probe struct {
		Flagged *bool                   `json:"flagged"`
		Score   *float64                `json:"score"`
		Frames  *[]VideoModerationFrame `json:"frames"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = VideoModerationResult(decoded)
	r.flaggedPresent = probe.Flagged != nil
	r.scorePresent = probe.Score != nil
	r.framesPresent = probe.Frames != nil
	return nil
}

// missingVerdictField names the decision field the body did not carry, or "".
func (r *VideoModerationResult) missingVerdictField() string {
	switch {
	case r == nil, !r.flaggedPresent:
		return "flagged"
	case !r.scorePresent:
		return "score"
	case !r.framesPresent:
		return "frames"
	}
	return ""
}

// HasVerdict reports whether this body carried a clip verdict this client can
// ACT ON — one backed by the per-frame evidence in the same body.
func (r *VideoModerationResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(unknownMediaBinding) == ""
}

// inconsistency describes why the clip verdict contradicts itself or the request
// that produced it, or "" when it is coherent. Mirrors moderateVideoFrames()
// exactly, so nothing a healthy engine emits is rejected.
func (r *VideoModerationResult) inconsistency(b mediaBinding) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	if r.Score < 0 || r.Score > 1 {
		return fmt.Sprintf("`score` is %v, outside the 0..1 range the engine clamps to", r.Score)
	}

	// COVERAGE — a verdict about 2 of the 40 frames you submitted is not a
	// verdict about the clip you asked about.
	if b.frameCount >= 0 {
		if r.FramesTotal != b.frameCount {
			return fmt.Sprintf("`framesTotal` is %d but %d frame(s) were submitted — the result does not "+
				"cover the clip this caller asked about", r.FramesTotal, b.frameCount)
		}
		if want := expectedSampledFrames(b.frameCount, b.sampleEveryN, b.maxFrames); r.FramesEvaluated != want {
			return fmt.Sprintf("`framesEvaluated` is %d but sampling every %d of %d frame(s) capped at %d "+
				"moderates %d — %d frame(s) were never looked at",
				r.FramesEvaluated, b.sampleEveryN, b.frameCount, b.maxFrames, want, want-r.FramesEvaluated)
		}
	}
	if r.FramesEvaluated < 1 {
		return "`framesEvaluated` is 0 — no frame was moderated, so `flagged` reports on nothing"
	}
	if len(r.Frames) != r.FramesEvaluated {
		return fmt.Sprintf("`framesEvaluated` is %d but `frames` carries %d per-frame result(s)",
			r.FramesEvaluated, len(r.Frames))
	}

	// DERIVATION — the clip verdict is an aggregate of the per-frame results in
	// the SAME body: flagged = ANY frame flagged, score = MAX frame score,
	// firstFlaggedFrame = index of the first. Re-deriving is what catches the
	// one-field edit that flips `flagged` to false on a body still reporting a
	// flagged frame — the same edit that turned an MCP "block" into "pass".
	wantFlagged := false
	wantScore := 0.0
	firstFlagged := -1
	var wantCategories []string
	for i, f := range r.Frames {
		if f.Index != i {
			return fmt.Sprintf("`frames[%d].index` is %d — the per-frame results are not the ordered "+
				"sample the aggregate was built from", i, f.Index)
		}
		if f.Score < 0 || f.Score > 1 {
			return fmt.Sprintf("`frames[%d].score` is %v, outside the 0..1 range the engine clamps to", i, f.Score)
		}
		if b.threshold >= 0 && !f.Flagged && f.Score >= b.threshold {
			return fmt.Sprintf("`frames[%d]` is not flagged but scores %v at threshold %v — each frame is "+
				"moderated by the same `score >= threshold` rule", i, f.Score, b.threshold)
		}
		if f.Score > wantScore {
			wantScore = f.Score
		}
		if f.Flagged {
			wantFlagged = true
			if firstFlagged < 0 {
				firstFlagged = i
			}
			wantCategories = append(wantCategories, f.Categories...)
		}
	}
	if r.Flagged != wantFlagged {
		return fmt.Sprintf("`flagged` is %t but %d of %d frame(s) are flagged — the clip verdict "+
			"contradicts the frames in the same body", r.Flagged, countFlaggedFrames(r.Frames), len(r.Frames))
	}
	if r.Score != wantScore {
		return fmt.Sprintf("`score` is %v but the worst frame scores %v — the clip score is the max "+
			"across moderated frames", r.Score, wantScore)
	}
	switch {
	case firstFlagged >= 0 && r.FirstFlaggedFrame == nil:
		return fmt.Sprintf("`firstFlaggedFrame` is absent but frame %d is flagged", firstFlagged)
	case firstFlagged >= 0 && *r.FirstFlaggedFrame != firstFlagged:
		return fmt.Sprintf("`firstFlaggedFrame` is %d but frame %d is the first flagged",
			*r.FirstFlaggedFrame, firstFlagged)
	case firstFlagged < 0 && r.FirstFlaggedFrame != nil:
		return fmt.Sprintf("`firstFlaggedFrame` is %d but no frame is flagged", *r.FirstFlaggedFrame)
	}
	if !sameStringSet(r.Categories, wantCategories) {
		return fmt.Sprintf("`categories` is %v but the flagged frames carry %v — the clip categories are "+
			"the union across flagged frames", r.Categories, wantCategories)
	}
	return ""
}

func countFlaggedFrames(frames []VideoModerationFrame) int {
	n := 0
	for _, f := range frames {
		if f.Flagged {
			n++
		}
	}
	return n
}

// DeepfakeLabelScore is one label/score pair from the forensic backend.
type DeepfakeLabelScore struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// MediaDeepfakeFrame is one scored frame inside a video MediaDeepfakeResult.
type MediaDeepfakeFrame struct {
	Index       int      `json:"index"`
	TimestampMs *float64 `json:"timestampMs,omitempty"`
	Synthetic   bool     `json:"synthetic"`
	Probability float64  `json:"probability"`
}

// mediaDeepfakeKinds is the CLOSED set the route tags the body with
// (apps/web/src/app/api/v1/moderation/deepfake/route.ts: `{ kind: "video", ... }`
// / `{ kind: "image", ... }`). Matched EXACTLY, same rule and same reasoning as
// mcpAuditVerdicts: a kind this client version does not recognise must DENY
// rather than be folded into the nearest one it does.
var mediaDeepfakeKinds = map[string]bool{"image": true, "video": true}

// MediaDeepfakeResult is the typed view of a POST /moderation/deepfake body
// (DeepfakeResult / VideoDeepfakeResult in packages/core/src/image/deepfake.ts,
// tagged with `kind` by the route).
//
// The worst instance of the class in this file: `Synthetic` zero-values to false
// and `Probability` to 0.0, which on a 0..1 synthetic-likelihood scale is
// "certainly authentic". Both readings of the natural gate —
// `if res.Synthetic { reject }` and `if res.Probability > threshold { reject }` —
// accepted every sample whose score never arrived.
type MediaDeepfakeResult struct {
	Kind                string               `json:"kind"`
	Synthetic           bool                 `json:"synthetic"`
	Probability         float64              `json:"probability"`
	MeanProbability     float64              `json:"meanProbability,omitempty"`
	Label               string               `json:"label,omitempty"`
	Scores              []DeepfakeLabelScore `json:"scores,omitempty"`
	FirstSyntheticFrame *int                 `json:"firstSyntheticFrame,omitempty"`
	FramesTotal         int                  `json:"framesTotal,omitempty"`
	FramesEvaluated     int                  `json:"framesEvaluated,omitempty"`
	Frames              []MediaDeepfakeFrame `json:"frames,omitempty"`
	Provider            string               `json:"provider,omitempty"`
	LatencyMs           float64              `json:"latencyMs,omitempty"`

	syntheticPresent   bool
	probabilityPresent bool
	meanPresent        bool
	framesPresent      bool
}

// UnmarshalJSON decodes the detection and records which decision fields the wire
// carried, including the video-only aggregates so the two shapes can be told
// apart by evidence rather than by the `kind` label alone.
func (r *MediaDeepfakeResult) UnmarshalJSON(data []byte) error {
	type alias MediaDeepfakeResult
	var probe struct {
		Synthetic       *bool                 `json:"synthetic"`
		Probability     *float64              `json:"probability"`
		MeanProbability *float64              `json:"meanProbability"`
		Frames          *[]MediaDeepfakeFrame `json:"frames"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = MediaDeepfakeResult(decoded)
	r.syntheticPresent = probe.Synthetic != nil
	r.probabilityPresent = probe.Probability != nil
	r.meanPresent = probe.MeanProbability != nil
	r.framesPresent = probe.Frames != nil
	return nil
}

// missingVerdictField names the decision field the body did not carry, or "".
func (r *MediaDeepfakeResult) missingVerdictField() string {
	switch {
	case r == nil, !r.syntheticPresent:
		return "synthetic"
	case !r.probabilityPresent:
		return "probability"
	case r.Kind == "video" && !r.framesPresent:
		return "frames"
	case r.Kind == "video" && !r.meanPresent:
		return "meanProbability"
	}
	return ""
}

// HasVerdict reports whether this body carried a deepfake verdict this client
// can ACT ON — a recognised `kind`, both decision fields, and (for a clip) the
// per-frame evidence the aggregate is derived from.
func (r *MediaDeepfakeResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(unknownMediaBinding) == ""
}

// inconsistency describes why the detection contradicts itself or the request
// that produced it, or "" when it is coherent. Mirrors detectImageDeepfake() /
// detectVideoDeepfake() exactly.
func (r *MediaDeepfakeResult) inconsistency(b mediaBinding) string {
	if r == nil {
		return "nil result"
	}
	if !mediaDeepfakeKinds[r.Kind] {
		return fmt.Sprintf("`kind` is %s, which is not one of image/video", quoteVerdict(r.Kind))
	}
	if b.kind != "" && r.Kind != b.kind {
		return fmt.Sprintf("`kind` is %q but this caller submitted a %s — the result is about "+
			"different media", r.Kind, b.kind)
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	if r.Probability < 0 || r.Probability > 1 {
		return fmt.Sprintf("`probability` is %v, outside the 0..1 range the engine clamps to", r.Probability)
	}

	if r.Kind == "image" {
		// DERIVATION — detectImageDeepfake() computes
		// `synthetic = probability >= threshold`, a pure comparison, so BOTH
		// directions are checked: a body claiming authentic at a suspicious
		// probability, and one claiming synthetic at a benign probability, are
		// each a body the engine could not have produced for this request.
		if b.threshold >= 0 {
			if want := r.Probability >= b.threshold; r.Synthetic != want {
				return fmt.Sprintf("`synthetic` is %t but probability %v against threshold %v derives %t — "+
					"the engine derives `synthetic` from `probability >= threshold`",
					r.Synthetic, r.Probability, b.threshold, want)
			}
		}
		return ""
	}

	// --- video ---
	if b.frameCount >= 0 {
		if r.FramesTotal != b.frameCount {
			return fmt.Sprintf("`framesTotal` is %d but %d frame(s) were submitted — the result does not "+
				"cover the clip this caller asked about", r.FramesTotal, b.frameCount)
		}
		if want := expectedSampledFrames(b.frameCount, b.sampleEveryN, b.maxFrames); r.FramesEvaluated != want {
			return fmt.Sprintf("`framesEvaluated` is %d but sampling every %d of %d frame(s) capped at %d "+
				"scores %d — %d frame(s) were never looked at",
				r.FramesEvaluated, b.sampleEveryN, b.frameCount, b.maxFrames, want, want-r.FramesEvaluated)
		}
	}
	if r.FramesEvaluated < 1 {
		return "`framesEvaluated` is 0 — no frame was scored, so `synthetic` reports on nothing"
	}
	if len(r.Frames) != r.FramesEvaluated {
		return fmt.Sprintf("`framesEvaluated` is %d but `frames` carries %d per-frame result(s)",
			r.FramesEvaluated, len(r.Frames))
	}

	wantSynthetic := false
	wantProb := 0.0
	sumProb := 0.0
	firstSynthetic := -1
	for i, f := range r.Frames {
		if f.Index != i {
			return fmt.Sprintf("`frames[%d].index` is %d — the per-frame results are not the ordered "+
				"sample the aggregate was built from", i, f.Index)
		}
		if f.Probability < 0 || f.Probability > 1 {
			return fmt.Sprintf("`frames[%d].probability` is %v, outside the 0..1 range the engine clamps to",
				i, f.Probability)
		}
		if b.threshold >= 0 {
			if want := f.Probability >= b.threshold; f.Synthetic != want {
				return fmt.Sprintf("`frames[%d].synthetic` is %t but probability %v against threshold %v "+
					"derives %t", i, f.Synthetic, f.Probability, b.threshold, want)
			}
		}
		sumProb += f.Probability
		if f.Probability > wantProb {
			wantProb = f.Probability
		}
		if f.Synthetic {
			wantSynthetic = true
			if firstSynthetic < 0 {
				firstSynthetic = i
			}
		}
	}
	if r.Synthetic != wantSynthetic {
		return fmt.Sprintf("`synthetic` is %t but the per-frame results say %t — the clip verdict "+
			"contradicts the frames in the same body", r.Synthetic, wantSynthetic)
	}
	if r.Probability != wantProb {
		return fmt.Sprintf("`probability` is %v but the worst frame scores %v — the clip probability is "+
			"the max across scored frames", r.Probability, wantProb)
	}
	// The mean is the one value that accumulates rounding, so it is the one
	// compared with slack.
	if wantMean := sumProb / float64(len(r.Frames)); math.Abs(r.MeanProbability-wantMean) > mediaFloatTolerance {
		return fmt.Sprintf("`meanProbability` is %v but the frames average %v", r.MeanProbability, wantMean)
	}
	switch {
	case firstSynthetic >= 0 && r.FirstSyntheticFrame == nil:
		return fmt.Sprintf("`firstSyntheticFrame` is absent but frame %d is synthetic", firstSynthetic)
	case firstSynthetic >= 0 && *r.FirstSyntheticFrame != firstSynthetic:
		return fmt.Sprintf("`firstSyntheticFrame` is %d but frame %d is the first synthetic",
			*r.FirstSyntheticFrame, firstSynthetic)
	case firstSynthetic < 0 && r.FirstSyntheticFrame != nil:
		return fmt.Sprintf("`firstSyntheticFrame` is %d but no frame is synthetic", *r.FirstSyntheticFrame)
	}
	return ""
}

// postGuardedMap runs one verdict-bearing POST and decodes the 2xx body
// TWICE: into the caller's map (the historical, unbroken return shape) and into
// verdict, the presence-tracking struct the refusal is decided on.
//
// Decoding the raw bytes rather than re-marshalling the map is deliberate — a
// map round-trip cannot tell an absent key from one explicitly set to null, and
// that distinction is the entire point of the presence probe.
func (c *Client) postGuardedMap(ctx context.Context, path string, body, verdict any) (map[string]any, error) {
	var raw json.RawMessage
	if err := c.doRequest(ctx, http.MethodPost, path, body, &raw); err != nil {
		return nil, err
	}
	// An empty body never reaches json.Unmarshal (doRequest skips it), which
	// would leave every presence flag false anyway — but it would also leave
	// verdict undecoded, so it is named explicitly rather than inferred.
	if len(raw) == 0 {
		return nil, nil
	}
	var result map[string]any
	if err := json.Unmarshal(raw, &result); err != nil {
		// A 2xx whose body is not even a JSON object (a bare array, a string, a
		// proxy's HTML error page) is not a verdict. Fall through with a nil map
		// and let the caller's presence check refuse.
		result = nil
	}
	if err := json.Unmarshal(raw, verdict); err != nil {
		return result, nil
	}
	return result, nil
}

// ModerationFrame is a single video frame for frame-by-frame moderation.
type ModerationFrame struct {
	ImageURL    string  `json:"imageUrl,omitempty"`
	ImageBase64 string  `json:"imageBase64,omitempty"`
	MimeType    string  `json:"mimeType,omitempty"`
	TimestampMs float64 `json:"timestampMs,omitempty"`
}

// ModerateImageRequest is the payload for ModerateImage. Provide either ImageURL
// (fetched server-side; SSRF-guarded) or ImageBase64 (inline).
type ModerateImageRequest struct {
	OrgID       string  `json:"orgId"`
	ProjectID   string  `json:"projectId"`
	ImageURL    string  `json:"imageUrl,omitempty"`
	ImageBase64 string  `json:"imageBase64,omitempty"`
	MimeType    string  `json:"mimeType,omitempty"`
	Threshold   float64 `json:"threshold,omitempty"`
	Provider    string  `json:"provider,omitempty"` // "openai"
}

// ModerateImage runs BYO vision-model content moderation on a single image.
// Requires a provider (OpenAI) key configured for the project.
// POST /api/v1/moderation/image.
//
// Fails CLOSED in the client: a 2xx that carries no `flagged`/`score` is not an
// allow, it is NO VERDICT, and this returns ErrCodeIndeterminate rather than a
// map whose missing keys assert to false and 0.0. Returning (nil, err) rather
// than the decoded map is deliberate — an indeterminate verdict must not be
// readable, or the zero value becomes the answer again at the next call site.
func (c *Client) ModerateImage(ctx context.Context, req *ModerateImageRequest) (map[string]any, error) {
	if req == nil || req.OrgID == "" || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateImage: orgID and projectID are required"}
	}
	if req.ImageURL == "" && req.ImageBase64 == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateImage: imageURL or imageBase64 is required"}
	}
	var verdict ImageModerationResult
	result, err := c.postGuardedMap(ctx, "/moderation/image", req, &verdict)
	if err != nil {
		return nil, fmt.Errorf("ModerateImage: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("ModerateImage", f, "POST /moderation/image")
	}
	// Bound to the REQUEST: the threshold is the one THIS caller asked for
	// (Threshold is `omitempty`, so a zero never reaches the wire and the server
	// applies its own default), so the response cannot restate it to escape the
	// derivation check.
	if reason := verdict.inconsistency(imageModerationBinding(req)); reason != "" {
		return nil, uninterpretableVerdict("ModerateImage", "POST /moderation/image", reason)
	}
	return result, nil
}

// imageModerationBinding is the request-side half of the image check.
func imageModerationBinding(req *ModerateImageRequest) mediaBinding {
	th := req.Threshold
	if th <= 0 {
		th = defaultVisionModerationThreshold
	}
	return mediaBinding{threshold: th, frameCount: -1}
}

// ModerateVideoRequest is the payload for ModerateVideo.
type ModerateVideoRequest struct {
	OrgID        string            `json:"orgId"`
	ProjectID    string            `json:"projectId"`
	Frames       []ModerationFrame `json:"frames"`
	Threshold    float64           `json:"threshold,omitempty"`
	MaxFrames    int               `json:"maxFrames,omitempty"`
	SampleEveryN int               `json:"sampleEveryN,omitempty"`
	Provider     string            `json:"provider,omitempty"`
}

// ModerateVideo runs BYO vision-model moderation across sampled video frames and
// aggregates the per-frame verdicts. POST /api/v1/moderation/video.
//
// Fails CLOSED in the client, on two axes: a 2xx with no `flagged`/`score`/
// `frames` is NO VERDICT, and a clip verdict that contradicts (or does not
// cover) the frames this caller submitted is one the gate must not read.
func (c *Client) ModerateVideo(ctx context.Context, req *ModerateVideoRequest) (map[string]any, error) {
	if req == nil || req.OrgID == "" || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateVideo: orgID and projectID are required"}
	}
	if len(req.Frames) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateVideo: at least one frame is required"}
	}
	var verdict VideoModerationResult
	result, err := c.postGuardedMap(ctx, "/moderation/video", req, &verdict)
	if err != nil {
		return nil, fmt.Errorf("ModerateVideo: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("ModerateVideo", f, "POST /moderation/video")
	}
	if reason := verdict.inconsistency(videoMediaBinding(
		defaultVisionModerationThreshold, req.Threshold, len(req.Frames), req.MaxFrames, req.SampleEveryN,
	)); reason != "" {
		return nil, uninterpretableVerdict("ModerateVideo", "POST /moderation/video", reason)
	}
	return result, nil
}

// videoMediaBinding is the request-side half of both frame-sampled checks. The
// sampling knobs are `omitempty`, so a zero never reaches the wire and the
// server applies its own default — which is what these fall back to.
func videoMediaBinding(defaultThreshold, threshold float64, frameCount, maxFrames, sampleEveryN int) mediaBinding {
	if threshold <= 0 {
		threshold = defaultThreshold
	}
	if maxFrames <= 0 {
		maxFrames = defaultMediaMaxFrames
	}
	if sampleEveryN <= 0 {
		sampleEveryN = defaultMediaSampleEveryN
	}
	return mediaBinding{
		threshold:    threshold,
		frameCount:   frameCount,
		maxFrames:    maxFrames,
		sampleEveryN: sampleEveryN,
		kind:         "video",
	}
}

// DetectMediaDeepfakeRequest is the payload for DetectMediaDeepfake. For a single
// image set ImageURL/ImageBase64; for a clip set Frames (Kind defaults from the
// shape of the input).
type DetectMediaDeepfakeRequest struct {
	OrgID        string            `json:"orgId"`
	ProjectID    string            `json:"projectId"`
	Kind         string            `json:"kind,omitempty"` // "image" | "video"
	ImageURL     string            `json:"imageUrl,omitempty"`
	ImageBase64  string            `json:"imageBase64,omitempty"`
	MimeType     string            `json:"mimeType,omitempty"`
	Frames       []ModerationFrame `json:"frames,omitempty"`
	Threshold    float64           `json:"threshold,omitempty"`
	MaxFrames    int               `json:"maxFrames,omitempty"`
	SampleEveryN int               `json:"sampleEveryN,omitempty"`
}

// DetectMediaDeepfake scores an image or video clip for AI-generated / deepfake
// likelihood via the operator-deployed deepfake backend. POST /api/v1/moderation/deepfake.
//
// Fails CLOSED in the client. On a 0..1 synthetic-likelihood scale the zero
// value is "certainly authentic", so BOTH readings of the natural gate
// (`if res["synthetic"].(bool)` and `if probability > threshold`) used to accept
// every sample whose score never arrived.
func (c *Client) DetectMediaDeepfake(ctx context.Context, req *DetectMediaDeepfakeRequest) (map[string]any, error) {
	if req == nil || req.OrgID == "" || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DetectMediaDeepfake: orgID and projectID are required"}
	}
	if req.ImageURL == "" && req.ImageBase64 == "" && len(req.Frames) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DetectMediaDeepfake: provide imageURL/imageBase64 or frames"}
	}
	var verdict MediaDeepfakeResult
	result, err := c.postGuardedMap(ctx, "/moderation/deepfake", req, &verdict)
	if err != nil {
		return nil, fmt.Errorf("DetectMediaDeepfake: %w", err)
	}
	if f := verdict.missingVerdictField(); f != "" {
		return nil, indeterminateVerdict("DetectMediaDeepfake", f, "POST /moderation/deepfake")
	}
	// Bound to the REQUEST on three axes: the threshold asked for, the frames
	// submitted, and the KIND — the binding re-derives which shape the route
	// will answer with, so a body tagged with the other one is a result about
	// different media however complete it looks.
	if reason := verdict.inconsistency(deepfakeMediaBinding(req)); reason != "" {
		return nil, uninterpretableVerdict("DetectMediaDeepfake", "POST /moderation/deepfake", reason)
	}
	return result, nil
}

// deepfakeMediaBinding is the request-side half of the deepfake check. It
// re-derives the kind the route will pick the SAME way the route does
// (`kind === "video" || (!imageUrl && !imageBase64 && !!frames)`), so a body
// tagged with the other shape is caught.
func deepfakeMediaBinding(req *DetectMediaDeepfakeRequest) mediaBinding {
	isVideo := req.Kind == "video" || (req.ImageURL == "" && req.ImageBase64 == "" && len(req.Frames) > 0)
	if !isVideo {
		th := req.Threshold
		if th <= 0 {
			th = defaultMediaDeepfakeThreshold
		}
		return mediaBinding{threshold: th, frameCount: -1, kind: "image"}
	}
	return videoMediaBinding(
		defaultMediaDeepfakeThreshold, req.Threshold, len(req.Frames), req.MaxFrames, req.SampleEveryN,
	)
}

// --- MCP / agent security ---

// McpAuditFinding is one finding from a pre-deploy MCP server audit.
type McpAuditFinding struct {
	Severity    string `json:"severity"`
	Category    string `json:"category"`
	Target      string `json:"target"`
	Title       string `json:"title"`
	Detail      string `json:"detail"`
	Remediation string `json:"remediation"`
}

// McpAuditReport is the severity-tiered result of a pre-deploy MCP server audit.
//
// Verdict is one of "block", "review", "pass" (route.ts). The zero value is the
// empty string, which is none of them — so a 200 carrying no verdict used to
// satisfy the natural gate `if report.Verdict == "block" { refuse }` and DEPLOY
// the server. AuditMcpServer now refuses instead; HasVerdict exposes the same
// check to anyone decoding a stored report themselves.
type McpAuditReport struct {
	Verdict   string            `json:"verdict"`
	RiskScore int               `json:"riskScore"`
	ToolCount int               `json:"toolCount"`
	Summary   map[string]int    `json:"summary"`
	Findings  []McpAuditFinding `json:"findings"`
}

// mcpAuditVerdicts is the CLOSED set of deploy verdicts the audit emits
// (packages/core/src/mcp-predeployment/index.ts:
// `verdict: "block" | "review" | "pass"`).
//
// Matched EXACTLY — no case folding, no trimming. The backend emits these three
// lowercase literals and nothing else, so "PASS" or "Block" on the wire is not a
// stylistic variation, it is evidence something between the auditor and this
// client rewrote the body. Folding "PASS" back to "pass" would DEPLOY on exactly
// that evidence; refusing costs a healthy server nothing because it never sends
// it. This is the allowlist rule the Python SDK adopted for firewall actions
// (evalguard/guardrails.py `_ALLOW_ACTIONS`): a verdict this client version does
// not recognise — a newer, stricter state such as "quarantine" — must DENY. An
// old client meeting a new restrictive verdict must not wave it through, and if
// the new verdict is actually a looser one the failure is VISIBLE (a blocked
// deploy) rather than silent.
var mcpAuditVerdicts = map[string]bool{"block": true, "review": true, "pass": true}

// mcpAuditSeverityWeight is the risk weighting the auditor applies
// (packages/core/src/mcp-predeployment/index.ts `SEVERITY_WEIGHT`), and its keys
// are the closed severity set a finding may carry.
var mcpAuditSeverityWeight = map[string]int{"critical": 40, "high": 20, "medium": 8, "low": 2}

// HasVerdict reports whether the audit returned a deploy verdict this client
// can ACT ON — i.e. one of block/review/pass, backed by the evidence in the
// same report.
//
// AUDIT 2026-08-03: this was `r.Verdict != ""`, a PRESENCE check standing in for
// a VALIDITY check. Any non-empty string satisfied it, so the natural gate
// `if report.Verdict == "block" { refuse }` deployed a server whose verdict was
// "Block", "quarantine", or "upstream timeout" — including one the audit had
// just scored 98/100.
//
// ROUND 2: validity against the closed set was still a check on a PROXY. The
// verdict is DERIVED — `summary.critical > 0 ? "block" : summary.high +
// summary.medium > 0 ? "review" : "pass"` — so flipping the one word "block" to
// "pass" on a report that still carries a critical finding was a valid verdict,
// and the deploy gate opened. HasVerdict now requires the verdict to agree with
// the findings and summary the report hands you.
//
// Consequence, deliberately: a report stripped down to `{"verdict":"block"}`
// has no verdict, because the auditor never emits a verdict without a summary.
// The asymmetry is the point — evidence-stripping can only ever turn a verdict
// into a refusal, never into a deploy.
func (r *McpAuditReport) HasVerdict() bool {
	return r != nil && mcpAuditVerdicts[r.Verdict] && r.inconsistency(-1) == ""
}

// inconsistency describes why the audit report contradicts itself or the request
// that produced it, or "" when it is coherent. toolCount is the number of tools
// submitted; pass -1 when unknown (decoding a stored report).
//
// Mirrors auditMcpServerConfig() exactly, so nothing a healthy auditor emits is
// rejected: `summary` tallies findings by severity with `total: findings.length`,
// `riskScore = min(100, Σ SEVERITY_WEIGHT[severity])`, `toolCount = tools.length`
// and the verdict is derived from the summary.
func (r *McpAuditReport) inconsistency(toolCount int) string {
	if r == nil {
		return "nil report"
	}
	if !mcpAuditVerdicts[r.Verdict] {
		return fmt.Sprintf("`verdict` is %s, which is not one of block/review/pass", quoteVerdict(r.Verdict))
	}

	// COVERAGE — an audit of 1 of the 3 tools you submitted is not an audit of
	// the server you are about to deploy.
	if toolCount >= 0 && r.ToolCount != toolCount {
		return fmt.Sprintf("`toolCount` is %d but %d tool(s) were submitted — the report does not "+
			"cover the server this caller asked about", r.ToolCount, toolCount)
	}

	// EVIDENCE — the auditor always emits a full summary alongside the verdict.
	// Requiring it is what stops a rewritten body from deleting the evidence to
	// escape the derivation check below.
	if r.Summary == nil {
		return "`summary` is absent — the report carries a verdict with no severity tally to justify it"
	}
	tally := map[string]int{}
	for n, f := range r.Findings {
		if _, ok := mcpAuditSeverityWeight[f.Severity]; !ok {
			return fmt.Sprintf("`findings[%d].severity` is %s, outside critical/high/medium/low",
				n, quoteVerdict(f.Severity))
		}
		tally[f.Severity]++
	}
	for sev := range mcpAuditSeverityWeight {
		got, ok := r.Summary[sev]
		if !ok {
			return fmt.Sprintf("`summary` has no %q count — the tally is incomplete", sev)
		}
		if got != tally[sev] {
			return fmt.Sprintf("`summary.%s` is %d but `findings` carries %d — the tally does not "+
				"match the evidence in the same report", sev, got, tally[sev])
		}
	}
	if total, ok := r.Summary["total"]; !ok || total != len(r.Findings) {
		return fmt.Sprintf("`summary.total` is %d but `findings` carries %d", r.Summary["total"], len(r.Findings))
	}

	// The two numbers a deploy gate actually reads are both DERIVED, so both are
	// recomputed from the findings rather than trusted.
	score := 0
	for _, f := range r.Findings {
		score += mcpAuditSeverityWeight[f.Severity]
	}
	if score > 100 {
		score = 100
	}
	if r.RiskScore != score {
		return fmt.Sprintf("`riskScore` is %d but the findings weigh %d — a `riskScore` deploy "+
			"threshold would be read against a number the report did not earn", r.RiskScore, score)
	}
	want := "pass"
	switch {
	case r.Summary["critical"] > 0:
		want = "block"
	case r.Summary["high"]+r.Summary["medium"] > 0:
		want = "review"
	}
	if r.Verdict != want {
		return fmt.Sprintf("`verdict` is %q but %d critical / %d high / %d medium finding(s) derive %q — "+
			"the deploy gate would read a verdict the evidence contradicts",
			r.Verdict, r.Summary["critical"], r.Summary["high"], r.Summary["medium"], want)
	}
	return ""
}

type mcpAuditBody struct {
	ProjectID string           `json:"projectId"`
	Server    map[string]any   `json:"server"`
	Tools     []map[string]any `json:"tools"`
}

// AuditMcpServer runs a pre-deploy security audit of an MCP server config.
// POST /api/v1/security/mcp-predeployment-audit.
func (c *Client) AuditMcpServer(ctx context.Context, projectID string, server map[string]any, tools []map[string]any) (*McpAuditReport, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "AuditMcpServer: projectID is required"}
	}
	if server == nil {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "AuditMcpServer: server is required"}
	}
	if tools == nil {
		tools = []map[string]any{}
	}
	var result McpAuditReport
	body := mcpAuditBody{ProjectID: projectID, Server: server, Tools: tools}
	if err := c.doRequest(ctx, http.MethodPost, "/security/mcp-predeployment-audit", body, &result); err != nil {
		return nil, fmt.Errorf("AuditMcpServer: %w", err)
	}
	if result.Verdict == "" {
		return nil, indeterminateVerdict("AuditMcpServer", "verdict",
			"POST /security/mcp-predeployment-audit")
	}
	// Bound to the REQUEST: len(tools) is what this caller submitted, so a
	// report that audited fewer cannot pass by restating its own coverage.
	if reason := result.inconsistency(len(tools)); reason != "" {
		return nil, uninterpretableVerdict("AuditMcpServer", "POST /security/mcp-predeployment-audit", reason)
	}
	return &result, nil
}

// AgentExecRedTeamResult is the breach verdict from an execution-layer red-team.
//
// Every numeric field zero-values to 0 and Verdict to "", so a 200 that is not
// a red-team result reads as "0 attacks, 0 breaches, no verdict" — a clean bill
// of health for a run that never happened. RunAgentExecRedTeam refuses instead.
type AgentExecRedTeamResult struct {
	TotalAttacks      int      `json:"totalAttacks"`
	DangerousAttempts int      `json:"dangerousAttempts"`
	Breaches          int      `json:"breaches"`
	Verdict           string   `json:"verdict"`
	Tools             []string `json:"tools"`
}

// agentExecVerdicts is the CLOSED set of red-team verdicts
// (packages/core/src/agent-exec-redteam/index.ts:
// `verdict: "breached" | "attempted" | "safe"`). Matched exactly, same rule and
// same reasoning as mcpAuditVerdicts.
var agentExecVerdicts = map[string]bool{"breached": true, "attempted": true, "safe": true}

// HasVerdict reports whether the red-team run returned a verdict this client
// can ACT ON — i.e. one of breached/attempted/safe, backed by the counts in the
// same result.
//
// AUDIT 2026-08-03: fixed alongside McpAuditReport.HasVerdict, which had the
// identical `!= ""` presence-for-validity defect. Leaving this one would have
// kept the same fail-open alive under a different field name: the natural gate
// `if res.Verdict == "breached" { fail the build }` reads "not breached" for
// "Breached", "BREACHED", or any verdict a newer server introduces.
//
// ROUND 2: this verdict is DERIVED too — `breaches > 0 ? "breached" :
// dangerousAttempts > 0 ? "attempted" : "safe"` — so the same one-word edit that
// turned an MCP "block" into "pass" turns "breached" into "safe" on a result
// still reporting breaches. Validity against the closed set does not see it;
// re-deriving from the counts does.
func (r *AgentExecRedTeamResult) HasVerdict() bool {
	return r != nil && agentExecVerdicts[r.Verdict] && r.inconsistency(-1) == ""
}

// inconsistency describes why the red-team result contradicts itself or the
// request that produced it, or "" when it is coherent. promptCount is the number
// of attack prompts the caller supplied; pass -1 (or 0, meaning "server's default
// set") when there is nothing to bind to.
//
// Mirrors runAgentExecRedTeam() exactly: `attacks` is one entry per prompt,
// `dangerousAttempts` and `breaches` are FILTERS of that same slice — so a
// breach is always also a dangerous attempt — and the verdict is derived from
// the two counts.
func (r *AgentExecRedTeamResult) inconsistency(promptCount int) string {
	if r == nil {
		return "nil result"
	}
	if !agentExecVerdicts[r.Verdict] {
		return fmt.Sprintf("`verdict` is %s, which is not one of breached/attempted/safe",
			quoteVerdict(r.Verdict))
	}
	// COVERAGE — "0 attacks, 0 breaches, safe" is a clean bill of health for a
	// run that never happened.
	if r.TotalAttacks < 1 {
		return "`totalAttacks` is 0 — no attack was run, so `verdict` reports on nothing"
	}
	if promptCount > 0 && r.TotalAttacks != promptCount {
		return fmt.Sprintf("`totalAttacks` is %d but %d attack prompt(s) were submitted — %d attack(s) "+
			"were never run", r.TotalAttacks, promptCount, promptCount-r.TotalAttacks)
	}
	if r.Breaches < 0 || r.DangerousAttempts < 0 {
		return fmt.Sprintf("negative counts (breaches=%d, dangerousAttempts=%d)", r.Breaches, r.DangerousAttempts)
	}
	// Every breached attack is by construction also an attempted one, and both
	// are subsets of the attacks that ran.
	if r.Breaches > r.DangerousAttempts {
		return fmt.Sprintf("`breaches` is %d but `dangerousAttempts` is only %d — every breach is a "+
			"dangerous attempt, so these counts cannot both be true", r.Breaches, r.DangerousAttempts)
	}
	if r.DangerousAttempts > r.TotalAttacks {
		return fmt.Sprintf("`dangerousAttempts` is %d but only %d attack(s) ran",
			r.DangerousAttempts, r.TotalAttacks)
	}
	want := "safe"
	switch {
	case r.Breaches > 0:
		want = "breached"
	case r.DangerousAttempts > 0:
		want = "attempted"
	}
	if r.Verdict != want {
		return fmt.Sprintf("`verdict` is %q but %d breach(es) / %d dangerous attempt(s) derive %q — "+
			"a build gate would read a verdict the counts contradict",
			r.Verdict, r.Breaches, r.DangerousAttempts, want)
	}
	return ""
}

type agentExecBody struct {
	ProjectID      string   `json:"projectId"`
	TargetProvider string   `json:"target_provider"`
	TargetModel    string   `json:"target_model"`
	AttackPrompts  []string `json:"attack_prompts,omitempty"`
}

// RunAgentExecRedTeam runs an execution-layer red-team against a target agent.
// POST /api/v1/security/agent-exec-redteam (uses the org's BYOK provider key).
func (c *Client) RunAgentExecRedTeam(ctx context.Context, projectID, provider, model string, attackPrompts []string) (*AgentExecRedTeamResult, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunAgentExecRedTeam: projectID is required"}
	}
	if provider == "" || model == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunAgentExecRedTeam: provider and model are required"}
	}
	var result AgentExecRedTeamResult
	body := agentExecBody{ProjectID: projectID, TargetProvider: provider, TargetModel: model, AttackPrompts: attackPrompts}
	if err := c.doRequest(ctx, http.MethodPost, "/security/agent-exec-redteam", body, &result); err != nil {
		return nil, fmt.Errorf("RunAgentExecRedTeam: %w", err)
	}
	if result.Verdict == "" {
		return nil, indeterminateVerdict("RunAgentExecRedTeam", "verdict",
			"POST /security/agent-exec-redteam")
	}
	// Bound to the REQUEST: when this caller supplied the attack prompts, the
	// run must have executed exactly that many.
	if reason := result.inconsistency(len(attackPrompts)); reason != "" {
		return nil, uninterpretableVerdict("RunAgentExecRedTeam", "POST /security/agent-exec-redteam", reason)
	}
	return &result, nil
}

// AgentCommEdge is one aggregated who-calls-whom edge.
type AgentCommEdge struct {
	From         string `json:"from"`
	To           string `json:"to"`
	CallCount    int    `json:"callCount"`
	ErrorCount   int    `json:"errorCount"`
	AvgLatencyMs int    `json:"avgLatencyMs"`
}

// AgentCommGraph is the agent-to-agent communication graph.
type AgentCommGraph struct {
	Services    []string        `json:"services"`
	Edges       []AgentCommEdge `json:"edges"`
	TotalCalls  int             `json:"totalCalls"`
	TotalErrors int             `json:"totalErrors"`
}

// GetAgentGraph fetches the agent-to-agent communication graph over a window.
// GET /api/v1/traces/graph.
func (c *Client) GetAgentGraph(ctx context.Context, projectID string, windowHours int) (*AgentCommGraph, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "GetAgentGraph: projectID is required"}
	}
	q := url.Values{}
	q.Set("projectId", projectID)
	if windowHours > 0 {
		q.Set("windowHours", fmt.Sprintf("%d", windowHours))
	}
	var result AgentCommGraph
	if err := c.doRequest(ctx, http.MethodGet, "/traces/graph?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetAgentGraph: %w", err)
	}
	return &result, nil
}

// --- Internal HTTP plumbing ---

// newIdempotencyKey returns a random RFC-4122 v4 UUID string. Used as the
// per-call Idempotency-Key so that retries of a non-idempotent POST/PUT/PATCH
// are deduplicated server-side (idempotency.ts keys on the `idempotency-key`
// header) instead of creating duplicate scans/runs and double-billing.
func newIdempotencyKey() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand should never fail; fall back to a time-seeded value so
		// retries within a single call still share one key (the goal here).
		nano := time.Now().UnixNano()
		for i := 0; i < 8; i++ {
			b[i] = byte(nano >> (8 * i))
		}
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// isUnsafeMethod reports whether a method mutates server state, so a retry
// must carry a stable Idempotency-Key to avoid duplicate side effects. GET
// and DELETE are naturally idempotent and need no key.
func isUnsafeMethod(method string) bool {
	switch method {
	case http.MethodPost, http.MethodPut, http.MethodPatch:
		return true
	default:
		return false
	}
}

// doRaw is the single place this client speaks HTTP. It runs the shared
// retry/backoff loop (transient 429/5xx retried with Retry-After honored, one
// stable Idempotency-Key reused across attempts of an unsafe method) and, on
// success, returns the raw response body and headers. Both doRequest (enveloped
// JSON) and FetchTraceAttachment (binary download) build on it, so every method
// shares one retry path and one typed *EvalGuardError surface. accept sets the
// Accept header ("application/json" for JSON routes, "application/octet-stream"
// for binary downloads).
func (c *Client) doRaw(ctx context.Context, method, path, accept string, body any) ([]byte, http.Header, error) {
	var bodyData []byte
	if body != nil {
		var err error
		bodyData, err = json.Marshal(body)
		if err != nil {
			return nil, nil, &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("failed to marshal request body: %v", err)}
		}
	}

	// Generate ONE Idempotency-Key per logical call (not per attempt) so the
	// retry loop below reuses it across every retry of an unsafe method. A
	// transient 502/network blip then dedups to a single server-side scan/run
	// instead of double-charging the customer.
	var idempotencyKey string
	if isUnsafeMethod(method) {
		idempotencyKey = newIdempotencyKey()
	}

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Wait before retrying. For 429 the server's Retry-After hint is
			// honoured but CLAMPED to maxRetryDelay — a hint may shorten a wait,
			// never extend it past the ceiling. Every branch is jittered so a
			// fleet that rate-limits together does not retry in lockstep and
			// re-stampede the recovering origin.
			delay := baseRetryDelay * time.Duration(math.Pow(2, float64(attempt-1)))
			if rateLimitErr, ok := lastErr.(*RateLimitError); ok {
				delay = rateLimitErr.RetryAfter
			}
			delay = jitteredDelay(delay)
			select {
			case <-ctx.Done():
				return nil, nil, &EvalGuardError{Code: ErrCodeTimeout, Message: "request cancelled while waiting to retry"}
			case <-time.After(delay):
			}
		}

		var bodyReader io.Reader
		if bodyData != nil {
			bodyReader = bytes.NewReader(bodyData)
		}

		req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bodyReader)
		if err != nil {
			return nil, nil, &EvalGuardError{Code: ErrCodeNetworkFailure, Message: fmt.Sprintf("failed to create request: %v", err)}
		}

		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", accept)
		req.Header.Set("User-Agent", userAgent)
		req.Header.Set("x-evalguard-client-version", clientVersion)
		if idempotencyKey != "" {
			// Same key on every attempt → server dedups the retry.
			req.Header.Set("Idempotency-Key", idempotencyKey)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			if ctx.Err() == context.DeadlineExceeded {
				return nil, nil, &EvalGuardError{Code: ErrCodeTimeout, Message: "request timed out"}
			}
			lastErr = &EvalGuardError{Code: ErrCodeNetworkFailure, Message: fmt.Sprintf("request failed: %v", err)}
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = &EvalGuardError{Code: ErrCodeNetworkFailure, Message: fmt.Sprintf("failed to read response body: %v", err)}
			continue
		}

		if resp.StatusCode >= 400 {
			requestID := resp.Header.Get("X-Request-ID")
			lastErr = c.handleErrorResponse(resp, respBody, requestID)
			// Retry on 429 (rate limit) and 5xx (server errors)
			if resp.StatusCode == 429 || resp.StatusCode >= 500 {
				continue
			}
			// Non-retryable client errors (401, 403, 404, 422, etc.)
			return nil, nil, lastErr
		}

		return respBody, resp.Header, nil
	}
	return nil, nil, lastErr
}

func (c *Client) doRequest(ctx context.Context, method, path string, body any, target any) error {
	respBody, header, err := c.doRaw(ctx, method, path, "application/json", body)
	if err != nil {
		return err
	}

	if target != nil && len(respBody) > 0 {
		if err := unmarshalEnvelope(respBody, target); err != nil {
			return &EvalGuardError{
				Code:      ErrCodeInternal,
				Message:   fmt.Sprintf("failed to decode response: %v", err),
				RequestID: header.Get("X-Request-ID"),
			}
		}
	}
	return nil
}

// unmarshalEnvelope decodes the standard EvalGuard API response envelope
// ({ "success": bool, "data": T }) into target by unwrapping "data". Every v1
// route replies through apiSuccess(data), so the typed result lives under
// "data" — unmarshalling the whole body into a typed struct left every field
// zero-valued. If the body has no "data" field (legacy / non-enveloped servers
// or a bare array), it falls back to decoding the whole body so callers still
// work. This is the single place the envelope is stripped; per-method handlers
// pass their plain result target.
func unmarshalEnvelope(body []byte, target any) error {
	var env struct {
		Data json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(body, &env); err == nil && len(env.Data) > 0 && string(env.Data) != "null" {
		return json.Unmarshal(env.Data, target)
	}
	return json.Unmarshal(body, target)
}

func (c *Client) handleErrorResponse(resp *http.Response, body []byte, requestID string) error {
	statusCode := resp.StatusCode
	// The EvalGuard API error envelope nests the reason under "error":
	//   { "success": false, "error": { "message": ..., "code": ... } }
	// (apiError in apps/web/src/lib/api.ts). Reading top-level "message"
	// therefore always came back empty → every error surfaced as the
	// generic HTTP status text. Read the nested field first, then fall
	// back to a top-level "message" (legacy / non-enveloped bodies).
	var apiErr struct {
		Error struct {
			Message string `json:"message"`
			Code    string `json:"code"`
		} `json:"error"`
		Message string `json:"message"`
		Code    string `json:"code"`
	}
	_ = json.Unmarshal(body, &apiErr)

	msg := apiErr.Error.Message
	if msg == "" {
		msg = apiErr.Message
	}
	if msg == "" {
		msg = http.StatusText(statusCode)
	}

	// The server also nests a structured machine code under "error".code
	// (VALIDATION_ERROR, INVALID_ID, DB_ERROR, …). Preserve it so callers can
	// branch on the real reason instead of a flattened HTTP category.
	serverCode := apiErr.Error.Code
	if serverCode == "" {
		serverCode = apiErr.Code
	}

	base := EvalGuardError{
		StatusCode: statusCode,
		Message:    msg,
		RequestID:  requestID,
	}

	switch {
	case statusCode == 401:
		base.Code = ErrCodeUnauthorized
		return &AuthError{EvalGuardError: base}
	case statusCode == 403:
		base.Code = ErrCodeForbidden
		return &AuthError{EvalGuardError: base}
	case statusCode == 404:
		base.Code = ErrCodeNotFound
		return &base
	case statusCode == 422:
		base.Code = ErrCodeValidation
		return &base
	case statusCode == 429:
		base.Code = ErrCodeRateLimit
		// Bounded, never verbatim — see maxRetryDelay.
		retryAfter := retryAfterFallback
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if seconds, parseErr := strconv.Atoi(strings.TrimSpace(ra)); parseErr == nil && seconds > 0 {
				retryAfter = time.Duration(seconds) * time.Second
			} else if at, dateErr := http.ParseTime(strings.TrimSpace(ra)); dateErr == nil {
				// HTTP-date form (RFC 9110 10.2.3): Atoi returns an error here,
				// so the hint used to be dropped on the floor.
				if d := time.Until(at); d > 0 {
					retryAfter = d
				}
			}
		}
		retryAfter = clampRetryDelay(retryAfter)
		return &RateLimitError{EvalGuardError: base, RetryAfter: retryAfter}
	default:
		// Every status the switch doesn't name explicitly used to collapse to
		// INTERNAL_ERROR, which mislabeled all 400 validation failures as server
		// faults and discarded the server's structured error.code. Prefer the
		// real server code when present; otherwise derive a sensible category
		// from the status class (4xx → validation-style client error, 5xx →
		// internal) so a 400 never reads as INTERNAL_ERROR.
		switch {
		case serverCode != "":
			base.Code = ErrorCode(serverCode)
		case statusCode >= 400 && statusCode < 500:
			base.Code = ErrCodeValidation
		default:
			base.Code = ErrCodeInternal
		}
		return &base
	}
}
