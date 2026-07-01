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
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"
)

const (
	maxRetries     = 3
	baseRetryDelay = 500 * time.Millisecond
)

const (
	DefaultBaseURL = "https://evalguard.ai/api/v1"
	DefaultTimeout = 30 * time.Second
	userAgent      = "evalguard-go/1.2.0"
	// clientVersion is sent as x-evalguard-client-version on every request so an
	// org that pins allowed client versions can enforce its policy on this SDK
	// (deep-audit 2026-06-21). Keep in lockstep with userAgent above.
	clientVersion = "1.2.0"
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
)

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

// WithBaseURL sets a custom API base URL.
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
}

// SecuritySeverityCounts is the per-severity finding tally returned by a scan.
type SecuritySeverityCounts struct {
	Critical int `json:"critical"`
	High     int `json:"high"`
	Medium   int `json:"medium"`
	Low      int `json:"low"`
}

// SecurityScanResult is the output of a synchronous security scan.
//
// POST /api/v1/security runs the scan inline and returns 201 with this
// summary (route.ts apiSuccess at lines 400-416): the scan row id, its
// terminal status, the aggregate safety score, the number of tests run,
// the wall-clock duration, the per-severity counts, and the total finding
// count. Individual findings are not inlined in this response — fetch them
// via the scan detail endpoint by ID.
type SecurityScanResult struct {
	ID             string                 `json:"id"`
	Status         string                 `json:"status"`
	Score          float64                `json:"score"`
	TotalTests     int                    `json:"totalTests"`
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
// Arize-parity registry: one row per (project, name, version), content-hash
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
func (c *Client) AnalyzeShadowAI(ctx context.Context, req *ShadowAIRequest) (*ShadowAIResult, error) {
	var result ShadowAIResult
	if err := c.doRequest(ctx, http.MethodPost, "/shadow-ai", req, &result); err != nil {
		return nil, fmt.Errorf("AnalyzeShadowAI: %w", err)
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
type CopilotAnalyzeRequest struct {
	Type     string           `json:"type"` // "security" or "eval"
	Model    string           `json:"model"`
	PassRate float64          `json:"passRate,omitempty"`
	Score    float64          `json:"score,omitempty"`
	Findings []map[string]any `json:"findings,omitempty"`
	Cases    []map[string]any `json:"cases,omitempty"`
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

// CheckCompliance runs a compliance check against frameworks.
func (c *Client) CheckCompliance(ctx context.Context, orgID string, frameworks []string) (map[string]any, error) {
	body := map[string]any{"orgId": orgID, "frameworks": frameworks}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/compliance/check", body, &result); err != nil {
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
type FirewallCheckResponse struct {
	Blocked     bool               `json:"blocked"`
	Score       float64            `json:"score"`
	Category    string             `json:"category,omitempty"`
	Subcategory string             `json:"subcategory,omitempty"`
	LatencyMs   float64            `json:"latencyMs"`
	Hits        []FirewallLayerHit `json:"hits,omitempty"`
}

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
	return &result, nil
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

// GetSecurityReport returns a security scan report.
func (c *Client) GetSecurityReport(ctx context.Context, scanID string) (map[string]any, error) {
	var result map[string]any
	q := url.Values{}
	q.Set("scanId", scanID)
	if err := c.doRequest(ctx, http.MethodGet, "/security/report?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetSecurityReport: %w", err)
	}
	return result, nil
}

// CodeScan scans code for security vulnerabilities.
func (c *Client) CodeScan(ctx context.Context, code, language, projectID string) (map[string]any, error) {
	body := map[string]any{"code": code, "language": language, "projectId": projectID}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/security/code-scan", body, &result); err != nil {
		return nil, fmt.Errorf("CodeScan: %w", err)
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

// CreateAnnotation creates an annotation on a log entry.
func (c *Client) CreateAnnotation(ctx context.Context, projectID, logID, label string) (map[string]any, error) {
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

// GetMarketplace returns the marketplace.
func (c *Client) GetMarketplace(ctx context.Context) (map[string]any, error) {
	var result map[string]any
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

// GenerateAISBOM generates an AI Software Bill of Materials.
func (c *Client) GenerateAISBOM(ctx context.Context, projectID string) (map[string]any, error) {
	body := map[string]any{"projectId": projectID}
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
func (c *Client) FetchTraceAttachment(ctx context.Context, traceID, attachmentID, projectID string) ([]byte, string, error) {
	q := url.Values{}
	q.Set("projectId", projectID)
	fullURL := c.baseURL + fmt.Sprintf("/traces/%s/attachments/%s?%s", url.PathEscape(traceID), url.PathEscape(attachmentID), q.Encode())
	reqHTTP, err := http.NewRequestWithContext(ctx, http.MethodGet, fullURL, nil)
	if err != nil {
		return nil, "", fmt.Errorf("FetchTraceAttachment: %w", err)
	}
	reqHTTP.Header.Set("Authorization", "Bearer "+c.apiKey)
	reqHTTP.Header.Set("Accept", "application/octet-stream")
	reqHTTP.Header.Set("User-Agent", userAgent)
	reqHTTP.Header.Set("x-evalguard-client-version", clientVersion)

	resp, err := c.httpClient.Do(reqHTTP)
	if err != nil {
		return nil, "", fmt.Errorf("FetchTraceAttachment: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, "", fmt.Errorf("FetchTraceAttachment: HTTP %d: %s", resp.StatusCode, string(body))
	}
	bytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("FetchTraceAttachment: %w", err)
	}
	return bytes, resp.Header.Get("Content-Type"), nil
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
type DeepfakeScore struct {
	Probability float64 `json:"probability"`
	Model       string  `json:"model,omitempty"`
}

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
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/rag/ingest", req, &result); err != nil {
		return nil, fmt.Errorf("IngestRAGDocuments: %w", err)
	}
	return result, nil
}

// RAGInjectionDocument is one document/chunk to vet for embedded prompt injection.
type RAGInjectionDocument struct {
	Text   string `json:"text"`
	Source string `json:"source,omitempty"`
}

// RAGInjectionScanResult is the outcome of ScanRAGInjection.
type RAGInjectionScanResult struct {
	Scanned         int              `json:"scanned"`
	Clean           bool             `json:"clean"`
	PoisonedCount   int              `json:"poisonedCount"`
	PoisonedIndices []int            `json:"poisonedIndices"`
	Violations      []map[string]any `json:"violations"`
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
	return &result, nil
}

// --- Multimodal moderation (image / video / deepfake) ---

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
// Requires a provider (OpenAI) key configured for the project. Fails closed.
// POST /api/v1/moderation/image.
func (c *Client) ModerateImage(ctx context.Context, req *ModerateImageRequest) (map[string]any, error) {
	if req == nil || req.OrgID == "" || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateImage: orgID and projectID are required"}
	}
	if req.ImageURL == "" && req.ImageBase64 == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateImage: imageURL or imageBase64 is required"}
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/moderation/image", req, &result); err != nil {
		return nil, fmt.Errorf("ModerateImage: %w", err)
	}
	return result, nil
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
func (c *Client) ModerateVideo(ctx context.Context, req *ModerateVideoRequest) (map[string]any, error) {
	if req == nil || req.OrgID == "" || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateVideo: orgID and projectID are required"}
	}
	if len(req.Frames) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ModerateVideo: at least one frame is required"}
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/moderation/video", req, &result); err != nil {
		return nil, fmt.Errorf("ModerateVideo: %w", err)
	}
	return result, nil
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
func (c *Client) DetectMediaDeepfake(ctx context.Context, req *DetectMediaDeepfakeRequest) (map[string]any, error) {
	if req == nil || req.OrgID == "" || req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DetectMediaDeepfake: orgID and projectID are required"}
	}
	if req.ImageURL == "" && req.ImageBase64 == "" && len(req.Frames) == 0 {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "DetectMediaDeepfake: provide imageURL/imageBase64 or frames"}
	}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/moderation/deepfake", req, &result); err != nil {
		return nil, fmt.Errorf("DetectMediaDeepfake: %w", err)
	}
	return result, nil
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
type McpAuditReport struct {
	Verdict   string            `json:"verdict"`
	RiskScore int               `json:"riskScore"`
	ToolCount int               `json:"toolCount"`
	Summary   map[string]int    `json:"summary"`
	Findings  []McpAuditFinding `json:"findings"`
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
	return &result, nil
}

// AgentExecRedTeamResult is the breach verdict from an execution-layer red-team.
type AgentExecRedTeamResult struct {
	TotalAttacks      int      `json:"totalAttacks"`
	DangerousAttempts int      `json:"dangerousAttempts"`
	Breaches          int      `json:"breaches"`
	Verdict           string   `json:"verdict"`
	Tools             []string `json:"tools"`
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

func (c *Client) doRequest(ctx context.Context, method, path string, body any, target any) error {
	var bodyData []byte
	if body != nil {
		var err error
		bodyData, err = json.Marshal(body)
		if err != nil {
			return &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("failed to marshal request body: %v", err)}
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
			// Wait before retrying. For 429, use Retry-After; otherwise exponential backoff.
			delay := baseRetryDelay * time.Duration(math.Pow(2, float64(attempt-1)))
			if rateLimitErr, ok := lastErr.(*RateLimitError); ok {
				delay = rateLimitErr.RetryAfter
			}
			select {
			case <-ctx.Done():
				return &EvalGuardError{Code: ErrCodeTimeout, Message: "request cancelled while waiting to retry"}
			case <-time.After(delay):
			}
		}

		var bodyReader io.Reader
		if bodyData != nil {
			bodyReader = bytes.NewReader(bodyData)
		}

		req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bodyReader)
		if err != nil {
			return &EvalGuardError{Code: ErrCodeNetworkFailure, Message: fmt.Sprintf("failed to create request: %v", err)}
		}

		req.Header.Set("Authorization", "Bearer "+c.apiKey)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("User-Agent", userAgent)
		req.Header.Set("x-evalguard-client-version", clientVersion)
		if idempotencyKey != "" {
			// Same key on every attempt → server dedups the retry.
			req.Header.Set("Idempotency-Key", idempotencyKey)
		}

		resp, err := c.httpClient.Do(req)
		if err != nil {
			if ctx.Err() == context.DeadlineExceeded {
				return &EvalGuardError{Code: ErrCodeTimeout, Message: "request timed out"}
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

		requestID := resp.Header.Get("X-Request-ID")

		if resp.StatusCode >= 400 {
			lastErr = c.handleErrorResponse(resp, respBody, requestID)
			// Retry on 429 (rate limit) and 5xx (server errors)
			if resp.StatusCode == 429 || resp.StatusCode >= 500 {
				continue
			}
			// Non-retryable client errors (401, 403, 404, 422, etc.)
			return lastErr
		}

		if target != nil && len(respBody) > 0 {
			if err := unmarshalEnvelope(respBody, target); err != nil {
				return &EvalGuardError{
					Code:      ErrCodeInternal,
					Message:   fmt.Sprintf("failed to decode response: %v", err),
					RequestID: requestID,
				}
			}
		}
		return nil
	}
	return lastErr
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
		retryAfter := 60 * time.Second
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if seconds, parseErr := strconv.Atoi(ra); parseErr == nil && seconds > 0 {
				retryAfter = time.Duration(seconds) * time.Second
			}
		}
		return &RateLimitError{EvalGuardError: base, RetryAfter: retryAfter}
	default:
		base.Code = ErrCodeInternal
		return &base
	}
}
