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
	"time"
)

const (
	maxRetries     = 3
	baseRetryDelay = 500 * time.Millisecond
)

const (
	DefaultBaseURL = "https://evalguard.ai/api/v1"
	DefaultTimeout = 30 * time.Second
	userAgent      = "evalguard-go/1.1.0"
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

// Trace represents a single observability trace.
type Trace struct {
	ID         string         `json:"id"`
	ParentID   string         `json:"parent_id,omitempty"`
	Name       string         `json:"name"`
	Input      any            `json:"input"`
	Output     any            `json:"output"`
	Metadata   map[string]any `json:"metadata,omitempty"`
	StartTime  time.Time      `json:"start_time"`
	EndTime    time.Time      `json:"end_time"`
	DurationMs float64        `json:"duration_ms"`
	TokensIn   int            `json:"tokens_in"`
	TokensOut  int            `json:"tokens_out"`
	CostUSD    float64        `json:"cost_usd"`
}

// GetTracesRequest contains filters for listing traces.
type GetTracesRequest struct {
	StartTime time.Time `json:"start_time,omitempty"`
	EndTime   time.Time `json:"end_time,omitempty"`
	Model     string    `json:"model,omitempty"`
	Limit     int       `json:"limit,omitempty"`
	Offset    int       `json:"offset,omitempty"`
}

// GetTracesResponse is a paginated list of traces.
type GetTracesResponse struct {
	Traces     []Trace `json:"traces"`
	TotalCount int     `json:"total_count"`
	HasMore    bool    `json:"has_more"`
}

// CreateDatasetRequest contains parameters for creating a dataset.
type CreateDatasetRequest struct {
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Items       []DatasetItem     `json:"items,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
}

// DatasetItem represents a single row in a dataset.
type DatasetItem struct {
	Input          string `json:"input"`
	ExpectedOutput string `json:"expected_output,omitempty"`
	Context        string `json:"context,omitempty"`
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

// --- API methods ---

// RunEval starts an asynchronous evaluation run.
//
// POST /api/v1/evals creates the run and fires background execution,
// returning 201 immediately with the new run's ID and status "running".
// The run does NOT finish before this call returns — poll GetEval(ID)
// for status, score, and results.
func (c *Client) RunEval(ctx context.Context, req *RunEvalRequest) (*EvalRunStarted, error) {
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
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "ListEvals: projectID is required"}
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
	if req.ProjectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "RunSecurityScan: ProjectID is required"}
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

// GetTraces retrieves observability traces.
func (c *Client) GetTraces(ctx context.Context, req *GetTracesRequest) (*GetTracesResponse, error) {
	var result GetTracesResponse
	if err := c.doRequest(ctx, http.MethodGet, "/traces", req, &result); err != nil {
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
func (c *Client) ListEvaluators(ctx context.Context, projectID, name string) (map[string]any, error) {
	if projectID == "" {
		return nil, &EvalGuardError{Code: ErrCodeValidation, Message: "projectID is required"}
	}
	var result map[string]any
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

// GetAuditLogs returns audit logs for an organization.
func (c *Client) GetAuditLogs(ctx context.Context, orgID string) ([]map[string]any, error) {
	var result []map[string]any
	q := url.Values{}
	q.Set("orgId", orgID)
	if err := c.doRequest(ctx, http.MethodGet, "/audit-logs?"+q.Encode(), nil, &result); err != nil {
		return nil, fmt.Errorf("GetAuditLogs: %w", err)
	}
	return result, nil
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

// GetCompliance returns compliance status.
func (c *Client) GetCompliance(ctx context.Context, orgID string) (map[string]any, error) {
	var result map[string]any
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

// ListNotifications returns user notifications.
func (c *Client) ListNotifications(ctx context.Context) ([]map[string]any, error) {
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/notifications", nil, &result); err != nil {
		return nil, fmt.Errorf("ListNotifications: %w", err)
	}
	return result, nil
}

// ListTemplates returns available eval templates.
func (c *Client) ListTemplates(ctx context.Context) ([]map[string]any, error) {
	var result []map[string]any
	if err := c.doRequest(ctx, http.MethodGet, "/templates", nil, &result); err != nil {
		return nil, fmt.Errorf("ListTemplates: %w", err)
	}
	return result, nil
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

// DetectDrift compares two eval runs for drift.
func (c *Client) DetectDrift(ctx context.Context, baselineRunID, currentRunID string) (map[string]any, error) {
	body := map[string]any{"baselineRunId": baselineRunID, "currentRunId": currentRunID}
	var result map[string]any
	if err := c.doRequest(ctx, http.MethodPost, "/drift/detect", body, &result); err != nil {
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

// ListPipelines returns all pipelines.
func (c *Client) ListPipelines(ctx context.Context) ([]map[string]any, error) {
	var result []map[string]any
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
