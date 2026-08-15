// Tests for the Phase 2 surface: named environments + Tools.

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

type capturedReq struct {
	method string
	path   string
	query  string
	body   map[string]any
}

// newCaptureServer returns a server that records the last request and echoes an
// empty JSON object (envelope-wrapped) for every route.
func newCaptureServer(t *testing.T, cap *capturedReq) (*Client, func()) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.method = r.Method
		cap.path = r.URL.Path
		cap.query = r.URL.RawQuery
		cap.body = nil
		if b, _ := io.ReadAll(r.Body); len(b) > 0 {
			_ = json.Unmarshal(b, &cap.body)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		// List routes decode into a slice; everything else into a map. Return
		// the matching envelope shape so decoding succeeds either way.
		p := r.URL.Path
		// A route decodes into a slice when it lists things. Post-#1071 the
		// surface is FLAT, so the shape is decided by (method, path, query),
		// not by a nested suffix:
		//   /tools/env-vars      — always the full variable list (GET/POST/DELETE)
		//   GET /environments    — ListEnvironments
		//   GET /prompts/deployments — ListPromptEnvironments
		//   GET /tools/deployments   — ListToolEnvironments
		//   GET /tools               — ListTools / ListToolVersions, EXCEPT when
		//                              `version=` is present, which is GetTool
		//                              asking for one record.
		isArray := contains(p, "/tools/env-vars") ||
			(r.Method == http.MethodGet &&
				(p == "/environments" || p == "/prompts/deployments" || p == "/tools/deployments" ||
					(p == "/tools" && !contains(r.URL.RawQuery, "version="))))
		if isArray {
			_, _ = w.Write([]byte(`{"data":[]}`))
		} else {
			// A REALISTIC object payload, not `{}`. These tests assert what the
			// client SENDS (method, path, query, body) and used an empty object
			// for the response, which the 2026-08-09 payload gate now correctly
			// refuses as a 2xx that answers nothing. The assertions are
			// untouched; only the stub is now a body a route could really send.
			_, _ = w.Write([]byte(`{"data":{"id":"env_1","name":"eu-prod","projectId":"proj-1",` +
				`"environment":"staging","version":3,"config":{},"createdAt":"2026-01-01T00:00:00Z"}}`))
		}
	}))
	client, err := NewClient("eg_test", WithBaseURL(srv.URL), WithTimeout(5*time.Second))
	if err != nil {
		srv.Close()
		t.Fatalf("NewClient: %v", err)
	}
	return client, srv.Close
}

func TestPhase2Values(t *testing.T) {
	if len(EnvironmentTagValues) != 2 || EnvironmentTagValues[0] != "default" || EnvironmentTagValues[1] != "other" {
		t.Fatalf("EnvironmentTagValues mismatch: %v", EnvironmentTagValues)
	}
	if SeededEnvironments[0].Name != "production" || SeededEnvironments[0].Tag != "default" {
		t.Fatalf("SeededEnvironments[0] mismatch: %+v", SeededEnvironments[0])
	}
	if SeededEnvironments[1].Name != "staging" || SeededEnvironments[1].Tag != "other" {
		t.Fatalf("SeededEnvironments[1] mismatch: %+v", SeededEnvironments[1])
	}
}

func TestValidateToolConfig(t *testing.T) {
	if ok, _ := ValidateToolConfig(ToolConfig{SourceCode: "return 1"}); !ok {
		t.Fatal("source-code-only config should be valid")
	}
	if ok, _ := ValidateToolConfig(ToolConfig{Function: &ToolFunction{Name: "f", Description: "d"}}); !ok {
		t.Fatal("function config should be valid")
	}
	if ok, errs := ValidateToolConfig(ToolConfig{}); ok || len(errs) == 0 {
		t.Fatal("empty config should be invalid")
	}
}

func TestCreateEnvironment(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()

	if _, err := c.CreateEnvironment(context.Background(), "proj-1", "eu-prod", "other"); err != nil {
		t.Fatalf("CreateEnvironment: %v", err)
	}
	if cap.method != http.MethodPost || cap.path != "/environments" {
		t.Fatalf("unexpected request: %s %s", cap.method, cap.path)
	}
	if cap.body["name"] != "eu-prod" || cap.body["tag"] != "other" {
		t.Fatalf("unexpected body: %+v", cap.body)
	}

	if _, err := c.CreateEnvironment(context.Background(), "proj-1", "  ", ""); err == nil {
		t.Fatal("expected error for empty environment name")
	}
}

func TestSetPromptDeployment(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()

	if _, err := c.SetPromptDeployment(context.Background(), "proj-1", "greeter", "staging", 3); err != nil {
		t.Fatalf("SetPromptDeployment: %v", err)
	}
	if cap.method != http.MethodPost || cap.path != "/prompts/deployments" {
		t.Fatalf("unexpected request: %s %s", cap.method, cap.path)
	}
	if cap.body["name"] != "greeter" || cap.body["env"] != "staging" || cap.body["version"].(float64) != 3 {
		t.Fatalf("unexpected body: %+v", cap.body)
	}
}

// RemovePromptDeployment must fail WITHOUT a network call: the prompt
// deployments route exports GET/POST/PUT and no DELETE, so the old
// `DELETE /prompts/{name}/deployments` reached the api/v1 catch-all.
func TestRemovePromptDeploymentIsUnsupported(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()

	if _, err := c.RemovePromptDeployment(context.Background(), "proj-1", "greeter", "staging"); err == nil {
		t.Fatal("expected RemovePromptDeployment to fail — the API has no DELETE route")
	}
	if cap.method != "" || cap.path != "" {
		t.Fatalf("expected no request to be sent, got %s %s", cap.method, cap.path)
	}
}

func TestCreateToolValidatesAndForwards(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()

	cfg := ToolConfig{Function: &ToolFunction{Name: "get_weather", Description: "d"}}
	if _, err := c.CreateTool(context.Background(), "proj-1", "weather", cfg); err != nil {
		t.Fatalf("CreateTool: %v", err)
	}
	if cap.method != http.MethodPost || cap.path != "/tools" {
		t.Fatalf("unexpected request: %s %s", cap.method, cap.path)
	}
	fn := cap.body["config"].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Fatalf("unexpected config: %+v", cap.body["config"])
	}

	if _, err := c.CreateTool(context.Background(), "proj-1", "bad", ToolConfig{}); err == nil {
		t.Fatal("expected error for empty tool config")
	}
}

func TestToolDeploymentAndEnvVars(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()
	ctx := context.Background()

	if _, err := c.SetToolDeployment(ctx, "proj-1", "weather", "production", 1); err != nil {
		t.Fatalf("SetToolDeployment: %v", err)
	}
	if cap.path != "/tools/deployments" || cap.body["toolName"] != "weather" ||
		cap.body["env"] != "production" {
		t.Fatalf("unexpected deploy request: %s %+v", cap.path, cap.body)
	}

	if _, err := c.ListToolEnvironments(ctx, "proj-1", "weather"); err != nil {
		t.Fatalf("ListToolEnvironments: %v", err)
	}
	if cap.path != "/tools/deployments" || !contains(cap.query, "name=weather") {
		t.Fatalf("unexpected list-envs request: %s?%s", cap.path, cap.query)
	}

	if _, err := c.AddToolEnvironmentVariable(ctx, "proj-1", "weather", "API_KEY", "k1"); err != nil {
		t.Fatalf("AddToolEnvironmentVariable: %v", err)
	}
	if cap.method != http.MethodPost || cap.path != "/tools/env-vars" {
		t.Fatalf("unexpected add-var request: %s %s", cap.method, cap.path)
	}
	if cap.body["name"] != "weather" {
		t.Fatalf("unexpected add-var tool name: %+v", cap.body)
	}
	vars := cap.body["variables"].([]any)
	v := vars[0].(map[string]any)
	if v["name"] != "API_KEY" || v["value"] != "k1" {
		t.Fatalf("unexpected variable body: %+v", v)
	}

	if _, err := c.DeleteToolEnvironmentVariable(ctx, "proj-1", "weather", "API_KEY"); err != nil {
		t.Fatalf("DeleteToolEnvironmentVariable: %v", err)
	}
	if cap.method != http.MethodDelete || cap.path != "/tools/env-vars" ||
		!contains(cap.query, "name=weather") || !contains(cap.query, "varName=API_KEY") {
		t.Fatalf("unexpected delete-var request: %s %s?%s", cap.method, cap.path, cap.query)
	}

	if _, err := c.AddToolEnvironmentVariable(ctx, "proj-1", "weather", "  ", "v"); err == nil {
		t.Fatal("expected error for empty env var name")
	}
}

func TestGetToolVersionQuery(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()

	if _, err := c.GetTool(context.Background(), "proj-1", "weather", 2); err != nil {
		t.Fatalf("GetTool: %v", err)
	}
	if cap.path != "/tools" {
		t.Fatalf("unexpected path: %s", cap.path)
	}
	if got := cap.query; got == "" || !contains(got, "version=2") || !contains(got, "name=weather") {
		t.Fatalf("expected name=weather&version=2 in query, got %q", got)
	}
}

// RemoveEnvironment and ListToolVersions/ListPromptEnvironments also moved to
// the flat surface in this change; assert each one's real wire shape.
func TestFlatEnvironmentAndVersionRoutes(t *testing.T) {
	var cap capturedReq
	c, cleanup := newCaptureServer(t, &cap)
	defer cleanup()
	ctx := context.Background()

	if _, err := c.RemoveEnvironment(ctx, "proj-1", "eu-prod"); err != nil {
		t.Fatalf("RemoveEnvironment: %v", err)
	}
	if cap.method != http.MethodDelete || cap.path != "/environments" ||
		!contains(cap.query, "name=eu-prod") {
		t.Fatalf("unexpected remove-env request: %s %s?%s", cap.method, cap.path, cap.query)
	}

	if _, err := c.ListToolVersions(ctx, "proj-1", "weather"); err != nil {
		t.Fatalf("ListToolVersions: %v", err)
	}
	if cap.path != "/tools" || !contains(cap.query, "name=weather") {
		t.Fatalf("unexpected list-versions request: %s?%s", cap.path, cap.query)
	}

	if _, err := c.ListPromptEnvironments(ctx, "proj-1", "greeter"); err != nil {
		t.Fatalf("ListPromptEnvironments: %v", err)
	}
	if cap.path != "/prompts/deployments" || !contains(cap.query, "name=greeter") {
		t.Fatalf("unexpected list-prompt-envs request: %s?%s", cap.path, cap.query)
	}

	if _, err := c.GetToolEnvironmentVariables(ctx, "proj-1", "weather"); err != nil {
		t.Fatalf("GetToolEnvironmentVariables: %v", err)
	}
	if cap.path != "/tools/env-vars" || !contains(cap.query, "name=weather") {
		t.Fatalf("unexpected get-vars request: %s?%s", cap.path, cap.query)
	}

	if _, err := c.RemoveToolDeployment(ctx, "proj-1", "weather", "production"); err != nil {
		t.Fatalf("RemoveToolDeployment: %v", err)
	}
	if cap.method != http.MethodDelete || cap.path != "/tools/deployments" ||
		!contains(cap.query, "name=weather") || !contains(cap.query, "env=production") {
		t.Fatalf("unexpected remove-deploy request: %s %s?%s", cap.method, cap.path, cap.query)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func hasSuffix(s, suf string) bool {
	return len(s) >= len(suf) && s[len(s)-len(suf):] == suf
}
