package evalguard

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// newJSONServer returns an httptest server asserting the path and capturing the
// request body, plus a *Client wired to it. Mirrors newCheckFirewallTestServer.
func newJSONServer(t *testing.T, wantPath string, status int, response any, capture *map[string]any) (*Client, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != wantPath {
			http.Error(w, "wrong path: "+r.URL.Path, http.StatusNotFound)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "wrong method: "+r.Method, http.StatusMethodNotAllowed)
			return
		}
		if capture != nil {
			body, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(body, capture)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	c, err := NewClient("eg_test", WithBaseURL(srv.URL), WithTimeout(5*time.Second))
	if err != nil {
		srv.Close()
		t.Fatalf("NewClient: %v", err)
	}
	return c, srv.Close
}

func TestChatCompletions_PostsAndReturnsRawOpenAIBody(t *testing.T) {
	var got map[string]any
	// The /chat/completions route returns the RAW OpenAI body (no {data} envelope).
	resp := map[string]any{
		"id": "chatcmpl-1", "model": "gpt-4o-mini",
		"choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": "pong"}}},
	}
	c, cleanup := newJSONServer(t, "/chat/completions", http.StatusOK, resp, &got)
	defer cleanup()

	r, err := c.ChatCompletions(context.Background(), &ChatCompletionsRequest{
		Model: "gpt-4o-mini", Messages: []map[string]any{{"role": "user", "content": "ping"}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	choices, ok := r["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("raw OpenAI body not passed through: %v", r)
	}
	if msg := choices[0].(map[string]any)["message"].(map[string]any)["content"]; msg != "pong" {
		t.Errorf("content: want pong, got %v", msg)
	}
	if got["stream"] != false {
		t.Errorf("stream must be forced false in body, got %v", got["stream"])
	}
	if got["model"] != "gpt-4o-mini" {
		t.Errorf("model not sent, got %v", got["model"])
	}
}

func TestChatCompletions_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.ChatCompletions(context.Background(), &ChatCompletionsRequest{Model: "", Messages: []map[string]any{{"role": "user"}}}); err == nil {
		t.Error("want error on empty model")
	}
	if _, err := c.ChatCompletions(context.Background(), &ChatCompletionsRequest{Model: "gpt-4o", Messages: nil}); err == nil {
		t.Error("want error on empty messages")
	}
}

func TestRunGuardrails_PostsTextAndUnwraps(t *testing.T) {
	var got map[string]any
	// A REAL /guardrails body. `severity` is not decoration: checkFirewall()
	// derives `action` from it (critical|high ⇒ block), so a fixture that omits
	// it is not a response the engine can produce — and the client now says so.
	resp := map[string]any{"data": map[string]any{
		"action": "block",
		"reasons": []map[string]any{
			{"rule": "pii-ssn", "type": "pii", "detail": "US SSN", "severity": "critical"},
		},
		"latencyMs": 1.42,
	}}
	c, cleanup := newJSONServer(t, "/guardrails", http.StatusOK, resp, &got)
	defer cleanup()

	r, err := c.RunGuardrails(context.Background(), "my ssn is 123-45-6789", "proj-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r["action"] != "block" {
		t.Errorf("action: want block, got %v", r["action"])
	}
	if got["text"] != "my ssn is 123-45-6789" {
		t.Errorf("text not sent, got %v", got["text"])
	}
	if got["projectId"] != "proj-1" {
		t.Errorf("projectId not sent, got %v", got["projectId"])
	}
}

func TestRunGuardrails_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.RunGuardrails(context.Background(), "", ""); err == nil {
		t.Error("want error on empty text")
	}
}

func TestScanSecrets_PostsContent(t *testing.T) {
	var got map[string]any
	// A REAL /security/secret-scan body: the route always emits scannedFiles,
	// filesWithFindings and the four-bucket severityCounts alongside the
	// findings, and they must agree with each other.
	resp := map[string]any{"data": map[string]any{
		"scannedFiles": 1.0, "filesWithFindings": 1.0, "findingsCount": 1.0,
		"findings": []map[string]any{{
			"ruleId": "aws-access-key-id", "description": "AWS access key id",
			"severity": "critical", "file": "input", "line": 3.0, "column": 11.0,
			"redactedMatch": "AKIA****************", "matchLength": 20.0,
		}},
		"severityCounts": map[string]any{"critical": 1.0, "high": 0.0, "medium": 0.0, "low": 0.0},
	}}
	c, cleanup := newJSONServer(t, "/security/secret-scan", http.StatusOK, resp, &got)
	defer cleanup()

	r, err := c.ScanSecrets(context.Background(), &SecretScanRequest{Content: "scan-this-content-blob"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r["findingsCount"] != 1.0 {
		t.Errorf("findingsCount: want 1, got %v", r["findingsCount"])
	}
	if got["content"] != "scan-this-content-blob" {
		t.Errorf("content not sent, got %v", got["content"])
	}
}

func TestScanSecrets_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.ScanSecrets(context.Background(), &SecretScanRequest{}); err == nil {
		t.Error("want error on empty content and files")
	}
}
