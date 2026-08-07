package evalguard

import (
	"context"
	"net/http"
	"testing"
)

func TestClassifyIntent(t *testing.T) {
	var got map[string]any
	c, cleanup := newJSONServer(t, "/governance/intent/classify", http.StatusOK,
		// A REAL classifier body. `risk` was never a field the route emits; the
		// six it does emit are below, and the full 13-entry score table is what
		// lets the client check that "harmful" was not downgraded on the way back.
		map[string]any{"data": map[string]any{
			"intent": "harmful", "confidence": 1.0, "sensitivity": "restricted",
			"riskScore": 1.0, "signals": []string{"intent:harmful:\"make a bomb\""},
			"scores": map[string]any{"harmful": 9.0, "code-generation": 0.0, "data-analysis": 0.0, "research": 0.0, "content-creation": 0.0, "financial": 0.0, "legal": 0.0, "hr-personnel": 0.0, "security-ops": 0.0, "customer-support": 0.0, "translation": 0.0, "summarization": 0.0, "general": 0.0},
		}}, &got)
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
		// A REAL lookup body: entries are 1:1 with the submitted purls IN ORDER,
		// and every summary counter is derived from them.
		map[string]any{"data": map[string]any{
			"entries": []map[string]any{{
				"purl": "pkg:npm/lodash@4.17.11", "status": "ok", "ecosystem": "npm",
				"name": "lodash", "version": "4.17.11",
				"vulnerabilities": []map[string]any{{"id": "GHSA-jf85-cpcp-j695", "cveId": "CVE-2019-10744"}},
			}},
			"summary": map[string]any{
				"total": 1.0, "queried": 1.0, "unsupported": 0.0, "invalid": 0.0,
				"vulnerable": 1.0, "vulnerabilitiesFound": 1.0,
			},
			"truncatedAdvisoryCount": 0.0,
		}}, &got)
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
		// A REAL iac-scan body. `findingsCount: 2` with no `findings` is exactly
		// the shape the fail-open produced, so it is no longer a valid fixture:
		// scannedFiles, findings and bySeverity all have to agree.
		map[string]any{"data": map[string]any{
			"scannedFiles": 1.0, "findingsCount": 2.0,
			"findings": []map[string]any{
				{"ruleId": "tf-s3-public-acl", "severity": "critical", "file": "main.tf",
					"line": 4.0, "title": "S3 bucket is public", "recommendation": "Set acl=private"},
				{"ruleId": "tf-sg-open-ssh", "severity": "high", "file": "main.tf",
					"line": 9.0, "title": "SSH open to 0.0.0.0/0", "recommendation": "Restrict CIDR"},
			},
			"bySeverity": map[string]any{"critical": 1.0, "high": 1.0, "medium": 0.0, "low": 0.0},
		}}, &got)
	defer cleanup()
	r, err := c.ScanIaC(context.Background(), []IaCFile{{Filename: "main.tf", Content: "resource x"}})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if r["findingsCount"] != 2.0 {
		t.Errorf("findingsCount: %v", r["findingsCount"])
	}
	if findings, _ := r["findings"].([]any); len(findings) != 2 {
		t.Errorf("findings not passed through: %v", r["findings"])
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
