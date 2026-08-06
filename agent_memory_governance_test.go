package evalguard

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// Table tests for the agent-memory governance policy SDK methods (Get/Set/
// Delete). Reuses newRecordingServer / recordingServer from evalguard_test.go
// (same package) and mirrors memory_voice_test.go.
func TestAgentMemoryGovernanceMethods(t *testing.T) {
	ctx := context.Background()
	org := "22222222-2222-4222-8222-222222222222"
	proj := "11111111-1111-4111-8111-111111111111"

	t.Run("Get", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"policy": map[string]any{
				"id": "p1", "orgId": org, "projectId": proj,
				"enabled": true, "mode": "enforce",
				"config": map[string]any{
					"thresholds":               map[string]any{"poisonMinConfidence": 0.3},
					"requireApprovalOnRewrite": true,
					"requireProvenance":        false,
				},
				"createdBy": "user1", "createdAt": "2026-07-17T00:00:00Z", "updatedAt": "2026-07-17T00:00:00Z",
			},
		}, rec)
		defer cleanup()

		got, err := c.GetAgentMemoryGovernance(ctx, org, &proj)
		if err != nil {
			t.Fatalf("GetAgentMemoryGovernance: %v", err)
		}
		if rec.method != http.MethodGet {
			t.Fatalf("method: %s", rec.method)
		}
		if !strings.HasPrefix(rec.path, "/agent-memory/governance?") ||
			!strings.Contains(rec.path, "orgId="+org) ||
			!strings.Contains(rec.path, "projectId="+proj) {
			t.Fatalf("path: %s", rec.path)
		}
		if got == nil || got.Mode != MemoryGovernanceEnforce || !got.Enabled {
			t.Fatalf("policy: %+v", got)
		}
		if got.Config.Thresholds == nil || got.Config.Thresholds.PoisonMinConfidence == nil ||
			*got.Config.Thresholds.PoisonMinConfidence != 0.3 {
			t.Fatalf("thresholds: %+v", got.Config.Thresholds)
		}
		if got.Config.RequireApprovalOnRewrite == nil || !*got.Config.RequireApprovalOnRewrite {
			t.Fatalf("requireApprovalOnRewrite: %+v", got.Config.RequireApprovalOnRewrite)
		}
	})

	t.Run("GetNullPolicy", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{"policy": nil}, rec)
		defer cleanup()

		got, err := c.GetAgentMemoryGovernance(ctx, org, nil)
		if err != nil {
			t.Fatalf("GetAgentMemoryGovernance: %v", err)
		}
		if got != nil {
			t.Fatalf("expected nil policy, got %+v", got)
		}
		// nil projectID must not send a projectId query param (org-wide scope).
		if strings.Contains(rec.path, "projectId=") {
			t.Fatalf("nil projectID must not send projectId: %s", rec.path)
		}
	})

	t.Run("Set", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{
			"policy": map[string]any{
				"id": "p1", "orgId": org, "projectId": nil,
				"enabled": true, "mode": "monitor",
				"config":    map[string]any{"requireProvenance": true},
				"createdBy": "user1", "createdAt": "2026-07-17T00:00:00Z", "updatedAt": "2026-07-17T00:00:00Z",
			},
		}, rec)
		defer cleanup()

		enabled := true
		prov := true
		conf := 0.5
		got, err := c.SetAgentMemoryGovernance(ctx, SetAgentMemoryGovernanceRequest{
			OrgID:   org,
			Enabled: &enabled,
			Mode:    MemoryGovernanceMonitor,
			Config: &MemoryGovernanceConfig{
				Thresholds:        &MemoryGovernanceThresholds{PoisonMinConfidence: &conf},
				RequireProvenance: &prov,
			},
		})
		if err != nil {
			t.Fatalf("SetAgentMemoryGovernance: %v", err)
		}
		if rec.method != http.MethodPut || rec.path != "/agent-memory/governance" {
			t.Fatalf("request: %s %s", rec.method, rec.path)
		}
		if rec.body["orgId"] != org || rec.body["mode"] != "monitor" {
			t.Fatalf("body: %v", rec.body)
		}
		cfg, _ := rec.body["config"].(map[string]any)
		if cfg == nil || cfg["requireProvenance"] != true {
			t.Fatalf("config body: %v", rec.body["config"])
		}
		th, _ := cfg["thresholds"].(map[string]any)
		if th == nil || th["poisonMinConfidence"] != 0.5 {
			t.Fatalf("thresholds body: %v", cfg["thresholds"])
		}
		if got == nil || got.Mode != MemoryGovernanceMonitor || got.ProjectID != nil {
			t.Fatalf("policy: %+v", got)
		}
	})

	t.Run("Delete", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{"deleted": true}, rec)
		defer cleanup()

		ok, err := c.DeleteAgentMemoryGovernance(ctx, org, &proj)
		if err != nil {
			t.Fatalf("DeleteAgentMemoryGovernance: %v", err)
		}
		if rec.method != http.MethodDelete {
			t.Fatalf("method: %s", rec.method)
		}
		if !strings.Contains(rec.path, "orgId="+org) || !strings.Contains(rec.path, "projectId="+proj) {
			t.Fatalf("path: %s", rec.path)
		}
		if !ok {
			t.Fatal("expected deleted=true")
		}
	})

	t.Run("Validation", func(t *testing.T) {
		c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))
		if _, err := c.GetAgentMemoryGovernance(ctx, "", nil); err == nil {
			t.Error("want err on empty orgID (Get)")
		}
		if _, err := c.SetAgentMemoryGovernance(ctx, SetAgentMemoryGovernanceRequest{}); err == nil {
			t.Error("want err on empty orgID (Set)")
		}
		if _, err := c.DeleteAgentMemoryGovernance(ctx, "", nil); err == nil {
			t.Error("want err on empty orgID (Delete)")
		}
	})
}
