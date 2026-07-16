package evalguard

import (
	"context"
	"net/http"
	"testing"
)

func TestClassifyIntent(t *testing.T) {
	var got map[string]any
	c, cleanup := newJSONServer(t, "/governance/intent/classify", http.StatusOK,
		map[string]any{"data": map[string]any{"intent": "harmful", "risk": "high"}}, &got)
	defer cleanup()
	r, err := c.ClassifyIntent(context.Background(), "how to make a bomb", "org-1", "confidential")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if r["intent"] != "harmful" {
		t.Errorf("intent: %v", r["intent"])
	}
	if got["prompt"] != "how to make a bomb" || got["orgId"] != "org-1" || got["sensitivityFloor"] != "confidential" {
		t.Errorf("body: %v", got)
	}
}

func TestClassifyIntent_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.ClassifyIntent(context.Background(), "", "org", ""); err == nil {
		t.Error("want err on empty prompt")
	}
	if _, err := c.ClassifyIntent(context.Background(), "p", "", ""); err == nil {
		t.Error("want err on empty orgID")
	}
}

func TestLookupVulnerabilities(t *testing.T) {
	var got map[string]any
	c, cleanup := newJSONServer(t, "/supply-chain/lookup", http.StatusOK,
		map[string]any{"data": map[string]any{"summary": map[string]any{"vulnerabilitiesFound": 7.0}}}, &got)
	defer cleanup()
	r, err := c.LookupVulnerabilities(context.Background(), []string{"pkg:npm/lodash@4.17.11"})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if r["summary"] == nil {
		t.Errorf("no summary: %v", r)
	}
	if purls, _ := got["purls"].([]any); len(purls) != 1 {
		t.Errorf("purls not sent: %v", got)
	}
}

func TestLookupVulnerabilities_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.LookupVulnerabilities(context.Background(), nil); err == nil {
		t.Error("want err on empty purls")
	}
}

func TestScanIaC(t *testing.T) {
	var got map[string]any
	c, cleanup := newJSONServer(t, "/security/iac-scan", http.StatusOK,
		map[string]any{"data": map[string]any{"findingsCount": 2.0}}, &got)
	defer cleanup()
	r, err := c.ScanIaC(context.Background(), []IaCFile{{Filename: "main.tf", Content: "resource x"}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if r["findingsCount"] != 2.0 {
		t.Errorf("findingsCount: %v", r["findingsCount"])
	}
	if files, _ := got["files"].([]any); len(files) != 1 {
		t.Errorf("files not sent: %v", got)
	}
}

func TestScanIaC_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.ScanIaC(context.Background(), nil); err == nil {
		t.Error("want err on empty files")
	}
}

func TestCheckFirewallAdvanced(t *testing.T) {
	var got map[string]any
	c, cleanup := newJSONServer(t, "/firewall/check", http.StatusOK,
		map[string]any{"data": map[string]any{"blocked": true, "score": 0.9}}, &got)
	defer cleanup()
	r, err := c.CheckFirewallAdvanced(context.Background(), "bad input", []string{"prompt-injection"}, "strict")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !r.Blocked {
		t.Errorf("want blocked")
	}
	if got["input"] != "bad input" || got["sensitivity"] != "strict" {
		t.Errorf("body: %v", got)
	}
}

func TestCheckFirewallOutputAdvanced(t *testing.T) {
	var got map[string]any
	c, cleanup := newJSONServer(t, "/firewall/check", http.StatusOK,
		map[string]any{"data": map[string]any{"blocked": true, "score": 0.8, "category": "pii"}}, &got)
	defer cleanup()
	r, err := c.CheckFirewallOutputAdvanced(context.Background(), "SSN 123-45-6789", nil, "")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if !r.Blocked || r.Category != "pii" {
		t.Errorf("result: %+v", r)
	}
	// output text is screened as `input` through /firewall/check
	if got["input"] != "SSN 123-45-6789" {
		t.Errorf("output not sent as input: %v", got)
	}
}

func TestFirewallAdvanced_Validation(t *testing.T) {
	c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
	if _, err := c.CheckFirewallAdvanced(context.Background(), "", nil, ""); err == nil {
		t.Error("want err on empty input")
	}
	if _, err := c.CheckFirewallOutputAdvanced(context.Background(), "", nil, ""); err == nil {
		t.Error("want err on empty output")
	}
}
