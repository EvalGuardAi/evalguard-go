// Package evalguard provides a Go client for the EvalGuard API.
//
// Usage:
//
//	client, err := evalguard.NewClient("your-api-key",
//		evalguard.WithBaseURL("https://api.evalguard.ai"),
//		evalguard.WithTimeout(30*time.Second),
//	)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	result, err := client.RunEval(ctx, &evalguard.RunEvalRequest{
//		DatasetID: "ds_abc123",
//		Model:     "gpt-4",
//		Metrics:   []string{"accuracy", "toxicity", "relevance"},
//	})
package evalguard

import (
	"bytes"
	"context"
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
	DefaultBaseURL = "https://api.evalguard.ai/v1"
	DefaultTimeout = 30 * time.Second
	userAgent      = "evalguard-go/1.0.0"
)

// ErrorCode represents categorized API error codes.
type ErrorCode string

const (
	ErrCodeUnauthorized    ErrorCode = "UNAUTHORIZED"
	ErrCodeForbidden       ErrorCode = "FORBIDDEN"
	ErrCodeNotFound        ErrorCode = "NOT_FOUND"
	ErrCodeRateLimit       ErrorCode = "RATE_LIMITED"
	ErrCodeValidation      ErrorCode = "VALIDATION_ERROR"
	ErrCodeInternal        ErrorCode = "INTERNAL_ERROR"
	ErrCodeTimeout         ErrorCode = "TIMEOUT"
	ErrCodeNetworkFailure  ErrorCode = "NETWORK_FAILURE"
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

// RunEvalRequest contains parameters for starting an evaluation run.
type RunEvalRequest struct {
	DatasetID   string            `json:"dataset_id"`
	Model       string            `json:"model"`
	Metrics     []string          `json:"metrics"`
	PromptID    string            `json:"prompt_id,omitempty"`
	Parameters  map[string]any    `json:"parameters,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
	Concurrency int               `json:"concurrency,omitempty"`
}

// EvalResult represents the outcome of an evaluation run.
type EvalResult struct {
	ID          string            `json:"id"`
	Status      string            `json:"status"`
	DatasetID   string            `json:"dataset_id"`
	Model       string            `json:"model"`
	Metrics     map[string]float64 `json:"metrics"`
	SampleCount int               `json:"sample_count"`
	CreatedAt   time.Time         `json:"created_at"`
	FinishedAt  *time.Time        `json:"finished_at,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
	Error       string            `json:"error,omitempty"`
}

// ListEvalsRequest contains filters for listing evaluations.
type ListEvalsRequest struct {
	DatasetID string `json:"dataset_id,omitempty"`
	Model     string `json:"model,omitempty"`
	Status    string `json:"status,omitempty"`
	Limit     int    `json:"limit,omitempty"`
	Offset    int    `json:"offset,omitempty"`
}

// ListEvalsResponse is a paginated list of evaluation results.
type ListEvalsResponse struct {
	Evals      []EvalResult `json:"evals"`
	TotalCount int          `json:"total_count"`
	HasMore    bool         `json:"has_more"`
}

// SecurityScanRequest contains parameters for a security scan.
type SecurityScanRequest struct {
	Prompts     []string `json:"prompts"`
	Model       string   `json:"model,omitempty"`
	ScanTypes   []string `json:"scan_types,omitempty"`
	Severity    string   `json:"severity,omitempty"`
}

// SecurityFinding represents a single finding from a security scan.
type SecurityFinding struct {
	ID          string  `json:"id"`
	Type        string  `json:"type"`
	Severity    string  `json:"severity"`
	Description string  `json:"description"`
	Prompt      string  `json:"prompt"`
	Score       float64 `json:"score"`
	Remediation string  `json:"remediation,omitempty"`
}

// SecurityScanResult is the output of a security scan.
type SecurityScanResult struct {
	ID        string            `json:"id"`
	Status    string            `json:"status"`
	Findings  []SecurityFinding `json:"findings"`
	Summary   map[string]int    `json:"summary"`
	CreatedAt time.Time         `json:"created_at"`
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
	Name        string           `json:"name"`
	Description string           `json:"description,omitempty"`
	Items       []DatasetItem    `json:"items,omitempty"`
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

// RunEval starts an evaluation run and returns the result.
func (c *Client) RunEval(ctx context.Context, req *RunEvalRequest) (*EvalResult, error) {
	var result EvalResult
	if err := c.doRequest(ctx, http.MethodPost, "/evals", req, &result); err != nil {
		return nil, fmt.Errorf("RunEval: %w", err)
	}
	return &result, nil
}

// GetEval retrieves an evaluation by ID.
func (c *Client) GetEval(ctx context.Context, evalID string) (*EvalResult, error) {
	var result EvalResult
	if err := c.doRequest(ctx, http.MethodGet, "/evals/"+evalID, nil, &result); err != nil {
		return nil, fmt.Errorf("GetEval: %w", err)
	}
	return &result, nil
}

// ListEvals returns a paginated list of evaluations.
func (c *Client) ListEvals(ctx context.Context, req *ListEvalsRequest) (*ListEvalsResponse, error) {
	var result ListEvalsResponse
	if err := c.doRequest(ctx, http.MethodGet, "/evals", req, &result); err != nil {
		return nil, fmt.Errorf("ListEvals: %w", err)
	}
	return &result, nil
}

// RunSecurityScan starts a security scan against prompts.
func (c *Client) RunSecurityScan(ctx context.Context, req *SecurityScanRequest) (*SecurityScanResult, error) {
	var result SecurityScanResult
	if err := c.doRequest(ctx, http.MethodPost, "/security/scan", req, &result); err != nil {
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

// --- Shadow AI ---

// ShadowAIRequest contains parameters for shadow AI analysis.
type ShadowAIRequest struct {
	Input    string `json:"input"`
	Provider string `json:"provider"`
	Model    string `json:"model"`
}

// ShadowAIResult is the output of a shadow AI analysis.
type ShadowAIResult struct {
	Event              map[string]any `json:"event"`
	PIIDetails         map[string]any `json:"piiDetails"`
	SensitiveDetails   map[string]any `json:"sensitiveDataDetails"`
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
	OverallScore          int            `json:"overallScore"`
	TotalModels           int            `json:"totalModels"`
	CriticalModels        int            `json:"criticalModels"`
	TotalMisconfigurations int           `json:"totalMisconfigurations"`
	DataFlows             int            `json:"dataFlows"`
	CrossBorderFlows      int            `json:"crossBorderFlows"`
	RiskDistribution      map[string]int `json:"riskDistribution"`
	Recommendations       []string       `json:"recommendations"`
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
	ID                   string  `json:"id"`
	ModelName            string  `json:"model_name"`
	Provider             *string `json:"provider,omitempty"`
	DisplayName          *string `json:"display_name,omitempty"`
	InputPricePer1MUSD   float64 `json:"input_price_per_1m_usd"`
	OutputPricePer1MUSD  float64 `json:"output_price_per_1m_usd"`
	ContextWindow        *int    `json:"context_window,omitempty"`
	Notes                *string `json:"notes,omitempty"`
	ProjectID            *string `json:"project_id,omitempty"`
}

type ListModelsResponse struct {
	Models []ModelRegistryEntry `json:"models"`
	Total  int                  `json:"total"`
}

type UpsertModelRequest struct {
	OrgID                string  `json:"orgId"`
	ModelName            string  `json:"modelName"`
	InputPricePer1MUSD   float64 `json:"inputPricePer1mUsd"`
	OutputPricePer1MUSD  float64 `json:"outputPricePer1mUsd"`
	ProjectID            *string `json:"projectId,omitempty"`
	Provider             *string `json:"provider,omitempty"`
	DisplayName          *string `json:"displayName,omitempty"`
	ContextWindow        *int    `json:"contextWindow,omitempty"`
	Notes                *string `json:"notes,omitempty"`
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
	ID        string                 `json:"id"`
	SpanID    string                 `json:"span_id"`
	Name      string                 `json:"name"`
	MimeType  string                 `json:"mime_type"`
	SizeBytes int                    `json:"size_bytes"`
	Metadata  map[string]any         `json:"metadata"`
	CreatedAt string                 `json:"created_at"`
}

type ListAttachmentsResponse struct {
	Attachments []SpanAttachment `json:"attachments"`
	Total       int              `json:"total"`
}

type UploadAttachmentRequest struct {
	TraceID    string         `json:"-"`
	ProjectID  string         `json:"projectId"`
	SpanID     string         `json:"spanId"`
	Name       string         `json:"name"`
	MimeType   string         `json:"mimeType"`
	// Data is the raw bytes to attach. Encoded to base64 on the wire.
	Data       []byte         `json:"-"`
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

// --- Internal HTTP plumbing ---

func (c *Client) doRequest(ctx context.Context, method, path string, body any, target any) error {
	var bodyData []byte
	if body != nil {
		var err error
		bodyData, err = json.Marshal(body)
		if err != nil {
			return &EvalGuardError{Code: ErrCodeValidation, Message: fmt.Sprintf("failed to marshal request body: %v", err)}
		}
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
			if err := json.Unmarshal(respBody, target); err != nil {
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

func (c *Client) handleErrorResponse(resp *http.Response, body []byte, requestID string) error {
	statusCode := resp.StatusCode
	var apiErr struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	}
	_ = json.Unmarshal(body, &apiErr)

	msg := apiErr.Message
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
