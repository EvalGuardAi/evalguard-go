package evalguard

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

func getTestClient(t *testing.T) *Client {
	t.Helper()
	apiKey := os.Getenv("EVALGUARD_API_KEY")
	baseURL := os.Getenv("EVALGUARD_BASE_URL")
	if apiKey == "" {
		apiKey = "eg_test_key"
	}
	if baseURL == "" {
		baseURL = "http://localhost:3000/api/v1"
	}
	client, err := NewClient(apiKey, WithBaseURL(baseURL), WithTimeout(30*time.Second))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return client
}

func TestNewClient(t *testing.T) {
	// Empty key should fail
	_, err := NewClient("")
	if err == nil {
		t.Fatal("expected error for empty API key")
	}

	// Valid key should succeed
	c, err := NewClient("eg_test_123")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.baseURL != DefaultBaseURL {
		t.Fatalf("expected default base URL, got %s", c.baseURL)
	}
}

func TestWithOptions(t *testing.T) {
	c, err := NewClient("eg_test", WithBaseURL("https://custom.api.com/v1"), WithTimeout(60*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if c.baseURL != "https://custom.api.com/v1" {
		t.Fatalf("expected custom URL, got %s", c.baseURL)
	}
	if c.httpClient.Timeout != 60*time.Second {
		t.Fatalf("expected 60s timeout, got %v", c.httpClient.Timeout)
	}
}

func TestErrorTypes(t *testing.T) {
	// EvalGuardError
	err := &EvalGuardError{Code: ErrCodeUnauthorized, Message: "bad key", StatusCode: 401, RequestID: "req-123"}
	if err.Error() == "" {
		t.Fatal("error string should not be empty")
	}
	if err.Code != ErrCodeUnauthorized {
		t.Fatalf("expected UNAUTHORIZED, got %s", err.Code)
	}

	// AuthError
	authErr := &AuthError{EvalGuardError: *err}
	if authErr.StatusCode != 401 {
		t.Fatal("AuthError should have status 401")
	}

	// RateLimitError
	rlErr := &RateLimitError{EvalGuardError: EvalGuardError{Code: ErrCodeRateLimit, Message: "slow down"}, RetryAfter: 30 * time.Second}
	if rlErr.RetryAfter != 30*time.Second {
		t.Fatal("RateLimitError should have 30s retry")
	}

	// Error without RequestID
	err2 := &EvalGuardError{Code: ErrCodeInternal, Message: "oops", StatusCode: 500}
	if err2.Error() == "" {
		t.Fatal("error string should not be empty")
	}
}

func TestRequestTypes(t *testing.T) {
	// RunEvalRequest
	req := &RunEvalRequest{
		DatasetID: "ds_123",
		Model:     "gpt-4o",
		Metrics:   []string{"accuracy", "toxicity"},
	}
	if req.Model != "gpt-4o" {
		t.Fatal("model mismatch")
	}

	// SecurityScanRequest
	scanReq := &SecurityScanRequest{
		Prompts:   []string{"test prompt"},
		Model:     "gpt-4o",
		ScanTypes: []string{"injection", "jailbreak"},
	}
	if len(scanReq.Prompts) != 1 {
		t.Fatal("prompts mismatch")
	}

	// ShadowAIRequest
	shadowReq := &ShadowAIRequest{
		Input:    "My SSN is 123-45-6789",
		Provider: "openai",
		Model:    "gpt-4o",
	}
	if shadowReq.Provider != "openai" {
		t.Fatal("provider mismatch")
	}

	// CopilotAnalyzeRequest
	copilotReq := &CopilotAnalyzeRequest{
		Type:     "security",
		Model:    "gpt-4o",
		PassRate: 0.75,
		Findings: []map[string]any{
			{"type": "injection", "severity": "critical", "title": "test", "description": "test", "passed": false},
		},
	}
	if copilotReq.Type != "security" {
		t.Fatal("type mismatch")
	}

	// FormalVerifyRequest
	fvReq := &FormalVerifyRequest{
		Output: "2 + 3 = 5",
		Constraints: []map[string]any{
			{"type": "mathematical-correctness", "id": "m1"},
		},
	}
	if fvReq.Output != "2 + 3 = 5" {
		t.Fatal("output mismatch")
	}

	// CreateDatasetRequest
	dsReq := &CreateDatasetRequest{
		Name:        "test-dataset",
		Description: "test",
		Items: []DatasetItem{
			{Input: "hello", ExpectedOutput: "world"},
		},
	}
	if len(dsReq.Items) != 1 {
		t.Fatal("items mismatch")
	}
}

func TestAllMethodSignatures(t *testing.T) {
	// Verify all methods exist and have correct signatures
	c, _ := NewClient("eg_test")
	ctx := context.Background()

	// This test just verifies the methods compile and are callable
	// Each method returns an error because we're not connected to a real server
	methods := []struct {
		name string
		fn   func() error
	}{
		{"RunEval", func() error { _, err := c.RunEval(ctx, &RunEvalRequest{}); return err }},
		{"GetEval", func() error { _, err := c.GetEval(ctx, "x"); return err }},
		{"ListEvals", func() error { _, err := c.ListEvals(ctx, &ListEvalsRequest{}); return err }},
		{"RunSecurityScan", func() error { _, err := c.RunSecurityScan(ctx, &SecurityScanRequest{}); return err }},
		{"GetTraces", func() error { _, err := c.GetTraces(ctx, &GetTracesRequest{}); return err }},
		{"CreateDataset", func() error { _, err := c.CreateDataset(ctx, &CreateDatasetRequest{}); return err }},
		{"AnalyzeShadowAI", func() error { _, err := c.AnalyzeShadowAI(ctx, &ShadowAIRequest{}); return err }},
		{"GetAIPosture", func() error { _, err := c.GetAIPosture(ctx, "pid"); return err }},
		{"AnalyzeCopilot", func() error { _, err := c.AnalyzeCopilot(ctx, &CopilotAnalyzeRequest{}); return err }},
		{"GetGatewayHealth", func() error { _, err := c.GetGatewayHealth(ctx); return err }},
		{"GetGatewayStats", func() error { _, err := c.GetGatewayStats(ctx, "pid"); return err }},
		{"GetCost", func() error { _, err := c.GetCost(ctx, "pid", "30d"); return err }},
		{"GetCostForecast", func() error { _, err := c.GetCostForecast(ctx, "pid"); return err }},
		{"GetMonitoringAlerts", func() error { _, err := c.GetMonitoringAlerts(ctx, "pid"); return err }},
		{"GetMonitoringDrift", func() error { _, err := c.GetMonitoringDrift(ctx, "pid"); return err }},
		{"CheckCompliance", func() error { _, err := c.CheckCompliance(ctx, "oid", nil); return err }},
		{"CreatePrompt", func() error { _, err := c.CreatePrompt(ctx, "p", "n", "c", "m"); return err }},
		{"ListPrompts", func() error { _, err := c.ListPrompts(ctx, "pid"); return err }},
		{"ListFirewallRules", func() error { _, err := c.ListFirewallRules(ctx, "pid"); return err }},
		{"ListGuardrails", func() error { _, err := c.ListGuardrails(ctx, "pid"); return err }},
		{"SubmitTicket", func() error { _, err := c.SubmitTicket(ctx, "bug", "s", "d", "low"); return err }},
		{"GetThreatIntelligence", func() error { _, err := c.GetThreatIntelligence(ctx, "pid"); return err }},
		{"GetAISBOM", func() error { _, err := c.GetAISBOM(ctx, "pid"); return err }},
		{"ListTeam", func() error { _, err := c.ListTeam(ctx, "oid"); return err }},
		{"GetAuditLogs", func() error { _, err := c.GetAuditLogs(ctx, "oid"); return err }},
		{"FormalVerify", func() error { _, err := c.FormalVerify(ctx, &FormalVerifyRequest{}); return err }},
		{"Ask", func() error { _, err := c.Ask(ctx, "q", "pid"); return err }},
		{"GetLeaderboard", func() error { _, err := c.GetLeaderboard(ctx, ""); return err }},
	}

	for _, m := range methods {
		t.Run(m.name, func(t *testing.T) {
			err := m.fn()
			// All should fail with network error since we're not connected
			if err == nil {
				t.Logf("%s returned nil error (unexpected but not fatal)", m.name)
			}
		})
	}

	fmt.Printf("Go SDK: %d methods verified compilable and callable\n", len(methods))
}
