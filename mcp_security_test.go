package evalguard

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// Tests for the MCP/agent-security + A2A-graph SDK methods (2026-06-12).
// Reuses newRecordingServer / recordingServer from evalguard_test.go.

func TestMcpAgentSecurityMethods(t *testing.T) {
	ctx := context.Background()
	pid := "11111111-1111-4111-8111-111111111111"

	t.Run("AuditMcpServer", func(t *testing.T) {
		rec := &recordingServer{}
		// Backend-shaped report: auditMcpServerConfig() always emits a complete
		// summary, a riskScore that is Σ SEVERITY_WEIGHT over the findings, and
		// a verdict DERIVED from the summary. A "block" with an empty findings
		// array and a partial summary is not a report the auditor can produce.
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"verdict": "block", "riskScore": 40, "toolCount": 1,
			"summary": map[string]any{"critical": 1, "high": 0, "medium": 0, "low": 0, "total": 1},
			"findings": []any{map[string]any{
				"severity": "critical", "category": "tool-description-injection", "target": "tool:x",
				"title": "Prompt injection in tool description", "detail": "…", "remediation": "…",
			}},
		}, rec)
		defer cleanup()
		got, err := c.AuditMcpServer(ctx, pid, map[string]any{"id": "s", "authSchemes": []any{}}, []map[string]any{{"name": "x"}})
		if err != nil {
			t.Fatalf("AuditMcpServer: %v", err)
		}
		if rec.path != "/security/mcp-predeployment-audit" {
			t.Fatalf("path: %s", rec.path)
		}
		if got.Verdict != "block" {
			t.Fatalf("verdict: %s", got.Verdict)
		}
	})

	t.Run("AuditMcpServerValidation", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{}, rec)
		defer cleanup()
		if _, err := c.AuditMcpServer(ctx, pid, nil, nil); err == nil {
			t.Fatal("expected validation error for nil server")
		}
	})

	t.Run("RunAgentExecRedTeam", func(t *testing.T) {
		rec := &recordingServer{}
		// `breaches` is a SUBSET of `dangerousAttempts` in runAgentExecRedTeam
		// (both filter the same `attacks` slice), so a breach with zero
		// dangerous attempts is not a result the engine can produce.
		c, cleanup := newRecordingServer(t, http.StatusOK,
			map[string]any{"verdict": "breached", "breaches": 1, "dangerousAttempts": 2, "totalAttacks": 5}, rec)
		defer cleanup()
		got, err := c.RunAgentExecRedTeam(ctx, pid, "openai", "gpt-4o-mini", nil)
		if err != nil {
			t.Fatalf("RunAgentExecRedTeam: %v", err)
		}
		if rec.path != "/security/agent-exec-redteam" || got.Verdict != "breached" {
			t.Fatalf("path %s verdict %s", rec.path, got.Verdict)
		}
		if rec.body["target_provider"] != "openai" {
			t.Fatalf("target_provider: %v", rec.body["target_provider"])
		}
	})

	t.Run("GetAgentGraph", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"services": []any{"a", "b"}, "edges": []any{}, "totalCalls": 0, "totalErrors": 0,
		}, rec)
		defer cleanup()
		got, err := c.GetAgentGraph(ctx, pid, 168)
		if err != nil {
			t.Fatalf("GetAgentGraph: %v", err)
		}
		if !strings.Contains(rec.path, "/traces/graph") || !strings.Contains(rec.path, "windowHours=168") {
			t.Fatalf("path: %s", rec.path)
		}
		if len(got.Services) != 2 {
			t.Fatalf("services: %v", got.Services)
		}
	})
}
