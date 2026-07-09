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
		// A route decodes into a slice when it lists things: any GET on an
		// `.../environments` or bare `/tools` collection, `.../versions`, or any
		// `environment-variables` route (which always returns the full var list).
		isArray := contains(p, "environment-variables") || hasSuffix(p, "/versions") ||
			(r.Method == http.MethodGet && (hasSuffix(p, "environments") || p == "/tools"))
		if isArray {
			_, _ = w.Write([]byte(`{"data":[]}`))
		} else {
			_, _ = w.Write([]byte(`{"data":{}}`))
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
	if cap.method != http.MethodPost || cap.path != "/prompts/greeter/deployments" {
		t.Fatalf("unexpected request: %s %s", cap.method, cap.path)
	}
	if cap.body["environment"] != "staging" || cap.body["version"].(float64) != 3 {
		t.Fatalf("unexpected body: %+v", cap.body)
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
	if cap.path != "/tools/weather/deployments" || cap.body["environment"] != "production" {
		t.Fatalf("unexpected deploy request: %s %+v", cap.path, cap.body)
	}

	if _, err := c.ListToolEnvironments(ctx, "proj-1", "weather"); err != nil {
		t.Fatalf("ListToolEnvironments: %v", err)
	}
	if cap.path != "/tools/weather/environments" {
		t.Fatalf("unexpected list-envs path: %s", cap.path)
	}

	if _, err := c.AddToolEnvironmentVariable(ctx, "proj-1", "weather", "API_KEY", "k1"); err != nil {
		t.Fatalf("AddToolEnvironmentVariable: %v", err)
	}
	if cap.method != http.MethodPost || cap.path != "/tools/weather/environment-variables" {
		t.Fatalf("unexpected add-var request: %s %s", cap.method, cap.path)
	}
	v := cap.body["variable"].(map[string]any)
	if v["name"] != "API_KEY" || v["value"] != "k1" {
		t.Fatalf("unexpected variable body: %+v", v)
	}

	if _, err := c.DeleteToolEnvironmentVariable(ctx, "proj-1", "weather", "API_KEY"); err != nil {
		t.Fatalf("DeleteToolEnvironmentVariable: %v", err)
	}
	if cap.method != http.MethodDelete || cap.path != "/tools/weather/environment-variables/API_KEY" {
		t.Fatalf("unexpected delete-var request: %s %s", cap.method, cap.path)
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
	if cap.path != "/tools/weather" {
		t.Fatalf("unexpected path: %s", cap.path)
	}
	if got := cap.query; got == "" || !contains(got, "version=2") {
		t.Fatalf("expected version=2 in query, got %q", got)
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
