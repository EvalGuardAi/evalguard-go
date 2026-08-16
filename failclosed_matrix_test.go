package evalguard

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"
)

// THE CROSS-CUTTING FAIL-CLOSED GATE.
//
// The per-method tests in verdict_guards_test.go pair one "must not read as
// allow" case with one "real results still parse" control, which is the right
// shape for a single method but does not scale to 158 exported methods -- and
// the defect this file exists to catch is precisely the one that shows up on
// "whatever nobody got to". Every method that returns or influences an
// allow/deny decision is driven against a loopback listener under EVERY fault
// mode below, and the test FAILS if any of them reads as safe.
//
// Adding decision method #159 means adding ONE line to matrixProbes(). That is
// the whole point: the sweep is a fixture, not a one-off audit.
//
// TWO ASSERTIONS, deliberately, because either alone is worthless:
//
//   1. CONTROL -- a well-formed CLEAN answer for each route MUST be ACCEPTED
//      (no error). Without it, "0 fail open" is indistinguishable from "every
//      probe is broken", because a probe that throws unconditionally "fails
//      closed" under every mode. Not hypothetical: this file's first draft had
//      16 of 24 probes unable to read a clean verdict, and the Python sibling
//      reported a clean 0/16 while 10 of its 16 probes threw TypeError on every
//      mode including the clean one.
//   2. MATRIX -- no probe reads as safe under any fault mode.

type faultMode struct {
	name   string
	status int
	body   string
	dead   bool // connection refused
	hang   bool // server never answers -> client timeout
}

var faultModes = []faultMode{
	{name: "missing-verdict", status: 200, body: `{"success":true,"data":{"latencyMs":3,"id":"x"}}`},
	{name: "null-verdict", status: 200, body: `{"success":true,"data":{"blocked":null,"action":null,"flagged":null,"passed":null,"violations":null,"findings":null,"findingsCount":null,"transcript":null,"policy":null,"gaps":null,"drift":null}}`},
	{name: "stringly-false", status: 200, body: `{"success":true,"data":{"blocked":"false","action":"false","flagged":"false","passed":"false"}}`},
	{name: "zero-verdict", status: 200, body: `{"success":true,"data":{"blocked":0,"action":0,"flagged":0,"passed":0}}`},
	{name: "empty-object", status: 200, body: `{"success":true,"data":{}}`},
	{name: "204-no-body", status: 204, body: ``},
	{name: "200-empty-body", status: 200, body: ``},
	{name: "200-literal-null", status: 200, body: `null`},
	{name: "2xx-success-false", status: 200, body: `{"success":false,"error":{"code":"ENGINE_DOWN","message":"down"},"data":{"blocked":false,"flagged":false,"passed":true,"action":"allow","reasons":[],"findingsCount":0,"findings":[]}}`},
	{name: "4xx", status: 403, body: `{"success":false,"error":{"message":"nope"}}`},
	{name: "5xx", status: 500, body: `{"success":false,"error":{"message":"boom"}}`},
	{name: "malformed-json", status: 200, body: `{"success":true,"data":{"blocked":`},
	{name: "conn-refused", dead: true},
	{name: "timeout", hang: true},
}

type probe struct {
	name string
	// call returns readAsSafe: with err == nil, would a normal caller's gate
	// conclude "allowed / clean / compliant / no drift"?
	call func(c *Client) (bool, error)
	// cleanOverrides are merged over baseCleanData() to build the CLEAN answer
	// for THIS route. Per-route because the routes genuinely disagree on the
	// same key: POST /security requires `status` in passed|failed|pending while
	// POST /compliance/check requires it to be "compliant", so no single body
	// can be a clean answer for both.
	cleanOverrides map[string]any
	// validAnswerModes names fault modes whose PAYLOAD happens to be a
	// legitimate answer for this particular route, so reading as safe there is
	// correct rather than a fail-open. Each entry must carry a reason.
	validAnswerModes map[string]string
}

func matrixProbes() []probe {
	ctx := context.Background()
	pid := "11111111-1111-4111-8111-111111111111"
	return []probe{
		{name: "CheckFirewall", call: func(c *Client) (bool, error) {
			r, err := c.CheckFirewall(ctx, &FirewallCheckRequest{Input: "x"})
			return err == nil && r != nil && !r.Blocked, err
		}},
		{name: "CheckFirewallAdvanced", call: func(c *Client) (bool, error) {
			r, err := c.CheckFirewallAdvanced(ctx, "x", []string{"injection"}, "strict")
			return err == nil && r != nil && !r.Blocked, err
		}},
		{name: "CheckFirewallOutputAdvanced", call: func(c *Client) (bool, error) {
			r, err := c.CheckFirewallOutputAdvanced(ctx, "x", nil, "strict")
			return err == nil && r != nil && !r.Blocked, err
		}},
		{name: "RunGuardrails", call: func(c *Client) (bool, error) {
			r, err := c.RunGuardrails(ctx, "x", pid)
			return err == nil && r["action"] != "block", err
		}},
		{name: "ScanSecrets", call: func(c *Client) (bool, error) {
			r, err := c.ScanSecrets(ctx, &SecretScanRequest{Content: "AKIA...", Path: "f.txt"})
			n, _ := r["findingsCount"].(float64)
			return err == nil && n == 0, err
		}},
		{name: "ScanIaC", call: func(c *Client) (bool, error) {
			r, err := c.ScanIaC(ctx, []IaCFile{{Filename: "a.tf", Content: "x"}})
			n, _ := r["findingsCount"].(float64)
			return err == nil && n == 0, err
		}},
		{name: "CodeScan", call: func(c *Client) (bool, error) {
			r, err := c.CodeScan(ctx, "code", "python", pid)
			n, _ := r["findingsCount"].(float64)
			return err == nil && n == 0, err
		}},
		{name: "LookupVulnerabilities", call: func(c *Client) (bool, error) {
			// "Safe" is NOT "entries is empty" -- a clean lookup returns one
			// entry per purl, each with an empty vulns list. Safe means no CVE
			// was reported for anything submitted.
			r, err := c.LookupVulnerabilities(ctx, []string{"pkg:npm/lodash@4.17.20"})
			if err != nil {
				return false, err
			}
			entries, _ := r["entries"].([]any)
			for _, e := range entries {
				m, _ := e.(map[string]any)
				if v, _ := m["vulns"].([]any); len(v) > 0 {
					return false, nil
				}
			}
			return true, nil
		}, cleanOverrides: map[string]any{
			// Each entry carries its own per-purl `status` (ok/unsupported/
			// invalid) — an "unsupported" ecosystem is NOT a clean lookup.
			"entries": []any{map[string]any{
				"purl": "pkg:npm/lodash@4.17.20", "status": "ok", "vulns": []any{},
			}},
			// `summary.queried` must equal the number of purls actually
			// answered for, so a body cannot claim coverage it did not earn.
			"summary": map[string]any{
				"total": 1, "queried": 1, "critical": 0, "high": 0, "medium": 0, "low": 0,
			},
		}},
		{name: "ClassifyIntent", call: func(c *Client) (bool, error) {
			r, err := c.ClassifyIntent(ctx, "x", "org_1", "")
			s, _ := r["riskScore"].(float64)
			return err == nil && s < 0.5, err
		}},
		{name: "ModerateImage", call: func(c *Client) (bool, error) {
			r, err := c.ModerateImage(ctx, &ModerateImageRequest{OrgID: "org_1", ProjectID: pid, ImageURL: "https://e.test/a.png"})
			f, _ := r["flagged"].(bool)
			return err == nil && !f, err
		}},
		{name: "RunSecurityScan", call: func(c *Client) (bool, error) {
			r, err := c.RunSecurityScan(ctx, &SecurityScanRequest{ProjectID: pid, Model: "gpt-4o", Prompt: "p", AttackTypes: []string{"jailbreak"}})
			return err == nil && r != nil && r.FindingsCount == 0, err
		}, cleanOverrides: map[string]any{
			// POST /security's `status` ladder is passed|failed|pending, NOT the
			// compliance ladder in baseCleanData(). Two routes, one key name,
			// two closed sets — which is exactly why cleanOverrides exists.
			"status": "passed",
			// This route emits `totalTests` and `findingsCount` from the SAME
			// findings array, and a scan with totalTests == 0 is refused as "a
			// clean bill of health for a run that never happened". So a clean
			// answer here has BOTH equal and non-zero — which also means this
			// method can never read as safe via FindingsCount == 0, i.e. it is
			// structurally fail-closed on that gate.
			"totalTests": 1, "findingsCount": 1,
			// This route's `score` is the 0..100 posture score (the firewall's
			// is 0..1), and it must AGREE with `status`: the route sets
			// "passed" at 70 and above.
			"score": 100,
			"findings": []any{map[string]any{
				"id": "f1", "severity": "low", "attackType": "jailbreak", "passed": true,
			}},
		}},
		// -- the RELEASE.md "still open after 1.6.0" list ------------------
		{name: "GetSecurityReport", call: func(c *Client) (bool, error) {
			r, err := c.GetSecurityReport(ctx, "assess_1")
			f, _ := r["findings"].([]any)
			return err == nil && len(f) == 0, err
		}},
		{name: "FormalVerify", call: func(c *Client) (bool, error) {
			r, err := c.FormalVerify(ctx, &FormalVerifyRequest{Output: "o", Constraints: []map[string]any{{"k": "v"}}})
			v, _ := r["violations"].([]any)
			return err == nil && len(v) == 0, err
		}, cleanOverrides: map[string]any{
			// Here `passed`/`failed` are COUNTS that must sum to
			// totalConstraints, not the boolean `passed` the scan routes emit.
			"passed": 1, "failed": 0, "totalConstraints": 1, "verified": true,
			"results": []any{map[string]any{
				"constraint": "k", "satisfied": true, "passed": true,
			}},
		}},
		{name: "CheckCompliance", call: func(c *Client) (bool, error) {
			r, err := c.CheckCompliance(ctx, &ComplianceCheckRequest{OrgID: "org_1", Framework: "soc2", Model: "gpt-4o", Provider: "openai", SystemPrompt: "s", APIKey: "sk"})
			g, _ := r["gaps"].([]any)
			return err == nil && len(g) == 0, err
		}},
		{name: "PromoteModelScan", call: func(c *Client) (bool, error) {
			r, err := c.PromoteModelScan(ctx, "scan_1", PromoteModelScanOpts{ToEnv: "prod"})
			bl, _ := r["blocked"].(bool)
			return err == nil && !bl, err
		}},
		{name: "TranscribeVoice", call: func(c *Client) (bool, error) {
			r, err := c.TranscribeVoice(ctx, pid, "AAAA", "en")
			return err == nil && r != nil && r.Text == "", err
		}},
		{name: "GetEUAIAct", call: func(c *Client) (bool, error) {
			r, err := c.GetEUAIAct(ctx, "org_1")
			g, _ := r["gaps"].([]any)
			return err == nil && len(g) == 0, err
		}},
		{name: "GetComplianceGaps", call: func(c *Client) (bool, error) {
			r, err := c.GetComplianceGaps(ctx, "org_1", "soc2")
			g, _ := r["gaps"].([]any)
			return err == nil && len(g) == 0, err
		}},
		{name: "GetModelScanAttestation", call: func(c *Client) (bool, error) {
			r, err := c.GetModelScanAttestation(ctx, "scan_1")
			return err == nil && r["attestation"] == nil, err
		}},
		{name: "GetAgentMemoryGovernance", call: func(c *Client) (bool, error) {
			r, err := c.GetAgentMemoryGovernance(ctx, "org_1", nil)
			return err == nil && (r == nil || !r.Enabled), err
		}, validAnswerModes: map[string]string{
			// The route emits `policy` on EVERY response, using an explicit
			// null to mean "no policy configured" -- so `"policy":null` in the
			// shared null-verdict payload is a real answer for this route, not
			// a missing one, and (nil, nil) is the correct reading of it. The
			// genuine fault here is an ABSENT `policy`, which the
			// missing-verdict / empty-object modes cover and the guard refuses.
			"null-verdict": "`policy:null` is this route's explicit \"no policy configured\" answer",
		}},
		{name: "DetectDrift", call: func(c *Client) (bool, error) {
			r, err := c.DetectDrift(ctx, "run_a", "run_b")
			// The route's field is `hasDrift` (driftProbe), not `driftDetected`.
			d, _ := r["hasDrift"].(bool)
			return err == nil && !d, err
		}},
		{name: "AuditMcpServer", call: func(c *Client) (bool, error) {
			r, err := c.AuditMcpServer(ctx, pid, map[string]any{"url": "https://e.test/mcp"}, []map[string]any{{"name": "t"}})
			// The deploy-verdict set is block|review|pass (mcpAuditVerdicts);
			// "block" is the deny, so that is what a caller gates on.
			return err == nil && r != nil && r.Verdict != "block", err
		}, cleanOverrides: map[string]any{
			// `summary.total` is cross-checked against `findings` here, whereas
			// on /supply-chain/lookup it is cross-checked against `entries`.
			"summary": map[string]any{
				"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0,
			},
		}},
		{name: "ScanRAGInjection", call: func(c *Client) (bool, error) {
			r, err := c.ScanRAGInjection(ctx, pid, []RAGInjectionDocument{{Text: "doc"}}, "")
			return err == nil && r != nil && r.PoisonedCount == 0, err
		}},
		{name: "ScoreVoiceDeepfake", call: func(c *Client) (bool, error) {
			r, err := c.ScoreVoiceDeepfake(ctx, pid, "AAAA")
			return err == nil && r != nil && r.Probability < 0.5, err
		}},
	}
}

// baseCleanData is the shared part of a well-formed, CLEAN answer. Each probe
// merges its own cleanOverrides over it (see probe.cleanOverrides for why a
// single shared body cannot work).
func baseCleanData() map[string]any {
	return map[string]any{
		"blocked": false, "score": 0, "category": "", "hits": []any{}, "latencyMs": 1,
		"action": "allow", "reasons": []any{}, "allowed": true, "violations": []any{},
		"flagged": false, "categories": []any{}, "provider": "openai", "confidence": 0.9,
		"findingsCount": 0, "findings": []any{}, "scannedFiles": 1, "filesWithFindings": 0,
		"severityCounts": map[string]any{"critical": 0, "high": 0, "medium": 0, "low": 0},
		"bySeverity":     map[string]any{"critical": 0, "high": 0, "medium": 0, "low": 0},
		"intent":         "general", "sensitivity": "public", "riskScore": 0, "signals": []any{},
		// The CLOSED 13-value PromptIntent set (verdict_guards.go promptIntents):
		// the guard requires a score for EVERY category, because a partial table
		// cannot be checked against the reported `intent`.
		"scores": map[string]any{
			"code-generation": 0.1, "data-analysis": 0.1, "research": 0.1,
			"content-creation": 0.1, "financial": 0.1, "legal": 0.1,
			"hr-personnel": 0.1, "security-ops": 0.1, "customer-support": 0.1,
			"translation": 0.1, "summarization": 0.1, "harmful": 0.0,
			"general": 0.9,
		},
		"status": "compliant", "passed": true, "overallScore": 100,
		"gaps": []any{}, "metCount": 0, "partialCount": 0, "notMetCount": 0, "untestedCount": 0,
		"totalRequirements": 0, "requirementResults": []any{}, "byCategory": map[string]any{},
		"framework": "soc2", "vulnerabilities": []any{},
		"executiveSummary": map[string]any{
			"posture": "clean", "riskLevel": "low", "totalVulnerabilities": 0,
		},
		"assessmentId": "assess_1", "scanId": "scan_1", "id": "scan_1",
		"state": "completed", "scanRan": true, "queued": false, "cached": false,
		"totalTests": 1, "failed": 0,
		"decision": "promoted", "toEnv": "prod", "gateStatus": "promoted",
		"incidents":     []any{},
		"attestation":   map[string]any{"bomFormat": "CycloneDX"},
		"policy":        map[string]any{"enabled": false, "mode": "off", "orgId": "org_1"},
		"driftDetected": false, "hasDrift": false, "severity": "none", "metrics": []any{},
		"overallDelta": 0, "metricDeltas": []any{},
		"verdict": "pass", "clean": true, "poisonedCount": 0, "documents": []any{},
		"screened": 1, "scanned": 1, "toolCount": 1,
		"probability": 0, "isDeepfake": false,
		"text": "a transcript", "transcript": "a transcript", "words": []any{}, "durationMs": 10,
		"language": "python",
		// `summary.total` is cross-checked against `entries`, so the two must
		// agree — the guard's evidence rule, not a free-form blob.
		"summary": map[string]any{
			"total": 1, "critical": 0, "high": 0, "medium": 0, "low": 0,
		},
		"entries":          []any{map[string]any{"purl": "pkg:npm/lodash@4.17.20", "vulns": []any{}}},
		"verified":         true,
		"totalConstraints": 1, "satisfiedCount": 1,
		"results":     []any{map[string]any{"constraint": "k", "satisfied": true}},
		"assessments": []any{},
		"projectId":   "11111111-1111-4111-8111-111111111111", "orgId": "org_1",
	}
}

// cleanBodyFor renders the CLEAN envelope this probe's route would really send.
func cleanBodyFor(t *testing.T, p probe) string {
	t.Helper()
	data := baseCleanData()
	for k, v := range p.cleanOverrides {
		data[k] = v
	}
	body, err := json.Marshal(map[string]any{"success": true, "data": data})
	if err != nil {
		t.Fatalf("marshal clean body for %s: %v", p.name, err)
	}
	return string(body)
}

func newMatrixClient(t *testing.T, fm faultMode) (*Client, func()) {
	t.Helper()
	if fm.dead {
		// Port 1 on loopback: nothing listens, so this is a refused connection
		// rather than a timeout.
		c, err := NewClient("eg_t", WithBaseURL("http://127.0.0.1:1"), WithTimeout(1*time.Second))
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}
		return c, func() {}
	}
	if fm.hang {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(900 * time.Millisecond)
		}))
		c, err := NewClient("eg_t", WithBaseURL(srv.URL), WithTimeout(150*time.Millisecond))
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}
		return c, srv.Close
	}
	if fm.status >= 300 && fm.status < 400 {
		// A CROSS-HOST redirect, built as a REAL two-listener harness rather
		// than a bare status code (2026-08-12).
		//
		// The SDK no longer refuses every 3xx — production answers 308 on the
		// verdict route for a base URL with a trailing slash, and refusing that
		// hard-fails a live customer's guardrail. The rule is SAME-HOST-ONLY,
		// so a 3xx row only proves something if the hop is CROSS-host.
		//
		// `attacker` shares the hostname 127.0.0.1 with `origin` and differs
		// ONLY by port. That is the exact shape that forwarded the API key in
		// the 2026-08-10 measurement, because Go's shouldCopyHeaderOnRedirect
		// compares hostnames only. It answers 200 with this mode's body, so a
		// client that follows the hop reads a clean, well-shaped ALLOW and the
		// matrix goes RED for every verdict-bearing method at once. That is
		// what makes this row a gate that can actually fail rather than a
		// status code nobody interprets.
		attacker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			if fm.body != "" {
				_, _ = w.Write([]byte(fm.body))
			}
		}))
		origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Location", attacker.URL+"/evil")
			w.WriteHeader(fm.status)
		}))
		c, err := NewClient("eg_t", WithBaseURL(origin.URL), WithTimeout(3*time.Second))
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}
		return c, func() { origin.Close(); attacker.Close() }
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(fm.status)
		if fm.body != "" {
			_, _ = w.Write([]byte(fm.body))
		}
	}))
	c, err := NewClient("eg_t", WithBaseURL(srv.URL), WithTimeout(3*time.Second))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return c, srv.Close
}

// TestFailClosedMatrix_ControlCleanVerdictIsAccepted proves the probes are real
// and the guards do not false-positive. Every probe is driven against a
// well-formed CLEAN answer for its own route and MUST come back without an
// error.
//
// Why "no error" rather than "reads as safe": a few probes are deliberately
// inverted — for TranscribeVoice and GetModelScanAttestation the DANGEROUS
// reading is the empty/absent one, so a genuinely clean response reads as
// not-safe by design. "The guard accepted a legitimate response" is the
// property that actually distinguishes a working probe from a broken one, and
// it holds for every probe regardless of polarity.
//
// Without this test a green TestFailClosedMatrix would be worthless: an
// unconditionally-throwing probe "fails closed" under every mode. That is not
// hypothetical — this file's first draft had 16 of 24 probes unable to read a
// clean verdict, and the Python sibling reported a clean 0/16 while 10 of its
// 16 probes were throwing TypeError on every mode including the clean one.
func TestFailClosedMatrix_ControlCleanVerdictIsAccepted(t *testing.T) {
	for _, p := range matrixProbes() {
		p := p
		t.Run(p.name, func(t *testing.T) {
			body := cleanBodyFor(t, p)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(body))
			}))
			defer srv.Close()

			c, err := NewClient("eg_t", WithBaseURL(srv.URL), WithTimeout(3*time.Second))
			if err != nil {
				t.Fatalf("NewClient: %v", err)
			}
			if _, callErr := p.call(c); callErr != nil {
				t.Errorf("BROKEN PROBE / OVER-STRICT GUARD: %s REJECTED a well-formed CLEAN "+
					"verdict, so its fail-closed result in TestFailClosedMatrix proves nothing:\n  %v",
					p.name, callErr)
			}
		})
	}
}

// TestFailClosedMatrix is the gate: no decision method may read as
// allowed/clean/compliant under any fault mode.
func TestFailClosedMatrix(t *testing.T) {
	probes := matrixProbes()
	failOpen := map[string][]string{}

	for _, fm := range faultModes {
		for _, p := range probes {
			if why, ok := p.validAnswerModes[fm.name]; ok {
				t.Logf("skipping %s/%s: %s", p.name, fm.name, why)
				continue
			}
			c, cleanup := newMatrixClient(t, fm)
			if safe, _ := p.call(c); safe {
				failOpen[p.name] = append(failOpen[p.name], fm.name)
			}
			cleanup()
		}
	}

	names := make([]string, 0, len(failOpen))
	for k := range failOpen {
		names = append(names, k)
	}
	sort.Strings(names)
	if len(names) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "%d of %d decision methods FAIL OPEN — a response carrying no usable "+
			"verdict read as allowed/clean:\n", len(names), len(probes))
		for _, n := range names {
			fmt.Fprintf(&b, "  %-28s %s\n", n, strings.Join(failOpen[n], ", "))
		}
		t.Fatal(b.String())
	}
	t.Logf("all %d decision methods fail CLOSED across %d fault modes", len(probes), len(faultModes))
}
