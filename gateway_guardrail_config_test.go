package evalguard

import (
	"context"
	"net/http"
	"strings"
	"testing"
)

// Table tests for the gateway guardrail-config SDK methods (List/Upsert/Delete).
// Reuses newRecordingServer / recordingServer from evalguard_test.go (same
// package) and mirrors agent_memory_governance_test.go — the fresh canonical
// example this surface was modelled on.
func TestGatewayGuardrailConfigMethods(t *testing.T) {
	ctx := context.Background()
	org := "22222222-2222-4222-8222-222222222222"
	proj := "11111111-1111-4111-8111-111111111111"
	secret := "33333333-3333-4333-8333-333333333333"

	t.Run("List", func(t *testing.T) {
		rec := &recordingServer{}
		// GET returns the array of rows directly under the envelope's `data`.
		c, cleanup := newRecordingServer(t, http.StatusOK, []map[string]any{
			{
				"id": "g1", "org_id": org, "project_id": proj,
				"vendor": "data-not-instructions", "config": map[string]any{},
				"secret_ref": nil, "on_flag": "block",
				"check_request": true, "check_response": false, "tokenize_pii": false,
				"enabled": true, "priority": 100.0,
				"created_at": "2026-07-17T00:00:00Z", "updated_at": "2026-07-17T00:00:00Z",
			},
			{
				"id": "g2", "org_id": org, "project_id": proj,
				"vendor": "lakera", "config": map[string]any{"profile": "strict"},
				"secret_ref": secret, "on_flag": "redact",
				"check_request": true, "check_response": true, "tokenize_pii": true,
				"enabled": true, "priority": 200.0,
				"created_at": "2026-07-17T00:00:00Z", "updated_at": "2026-07-17T00:00:00Z",
			},
		}, rec)
		defer cleanup()

		got, err := c.ListGuardrailConfigs(ctx, proj)
		if err != nil {
			t.Fatalf("ListGuardrailConfigs: %v", err)
		}
		if rec.method != http.MethodGet {
			t.Fatalf("method: %s", rec.method)
		}
		if !strings.HasPrefix(rec.path, "/gateway/guardrails?") ||
			!strings.Contains(rec.path, "projectId="+proj) {
			t.Fatalf("path: %s", rec.path)
		}
		if len(got) != 2 {
			t.Fatalf("want 2 rows, got %d: %+v", len(got), got)
		}
		// Local vendor row: SecretRef nil, flag action typed.
		if got[0].Vendor != "data-not-instructions" || got[0].SecretRef != nil ||
			got[0].OnFlag != GuardrailFlagBlock || got[0].Priority != 100 {
			t.Fatalf("row0: %+v", got[0])
		}
		// Partner vendor row: SecretRef populated, tokenize + response check on.
		if got[1].Vendor != "lakera" || got[1].SecretRef == nil || *got[1].SecretRef != secret ||
			got[1].OnFlag != GuardrailFlagRedact || !got[1].TokenizePii || !got[1].CheckResponse {
			t.Fatalf("row1: %+v", got[1])
		}
	})

	t.Run("UpsertLocalVendor", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusCreated, map[string]any{
			"id": "g1", "org_id": org, "project_id": proj,
			"vendor": "tool-call-circuit-breaker", "config": map[string]any{"maxRepeats": 3.0},
			"secret_ref": nil, "on_flag": "flag",
			"check_request": false, "check_response": true, "tokenize_pii": false,
			"enabled": true, "priority": 100.0,
			"created_at": "2026-07-17T00:00:00Z", "updated_at": "2026-07-17T00:00:00Z",
		}, rec)
		defer cleanup()

		enabled := true
		got, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{
			OrgID:     org,
			ProjectID: proj,
			Vendor:    "tool-call-circuit-breaker",
			Config:    map[string]any{"maxRepeats": 3},
			OnFlag:    GuardrailFlagFlag,
			Enabled:   &enabled,
			// No SecretRef — a local vendor must omit it.
		})
		if err != nil {
			t.Fatalf("UpsertGuardrailConfig: %v", err)
		}
		if rec.method != http.MethodPost || rec.path != "/gateway/guardrails" {
			t.Fatalf("request: %s %s", rec.method, rec.path)
		}
		if rec.body["orgId"] != org || rec.body["projectId"] != proj ||
			rec.body["vendor"] != "tool-call-circuit-breaker" || rec.body["onFlag"] != "flag" {
			t.Fatalf("body: %v", rec.body)
		}
		// A local vendor must NOT send a secretRef key at all (omitempty + nil).
		if _, present := rec.body["secretRef"]; present {
			t.Fatalf("local vendor must not send secretRef: %v", rec.body)
		}
		if got == nil || got.Vendor != "tool-call-circuit-breaker" || got.SecretRef != nil ||
			got.OnFlag != GuardrailFlagFlag {
			t.Fatalf("result: %+v", got)
		}
	})

	t.Run("UpsertPartnerVendor", func(t *testing.T) {
		rec := &recordingServer{}
		c, cleanup := newRecordingServer(t, http.StatusCreated, map[string]any{
			"id": "g2", "org_id": org, "project_id": proj,
			"vendor": "aporia", "config": map[string]any{},
			"secret_ref": secret, "on_flag": "block",
			"check_request": true, "check_response": false, "tokenize_pii": false,
			"enabled": true, "priority": 100.0,
			"created_at": "2026-07-17T00:00:00Z", "updated_at": "2026-07-17T00:00:00Z",
		}, rec)
		defer cleanup()

		got, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{
			OrgID:     org,
			ProjectID: proj,
			Vendor:    "aporia",
			SecretRef: &secret,
		})
		if err != nil {
			t.Fatalf("UpsertGuardrailConfig: %v", err)
		}
		if rec.body["secretRef"] != secret {
			t.Fatalf("partner vendor must send secretRef: %v", rec.body)
		}
		if got == nil || got.SecretRef == nil || *got.SecretRef != secret {
			t.Fatalf("result: %+v", got)
		}
	})

	t.Run("Delete", func(t *testing.T) {
		rec := &recordingServer{}
		// DELETE returns { deleted: "<id>" } (the id string, not a bool).
		c, cleanup := newRecordingServer(t, http.StatusOK, map[string]any{"deleted": "g1"}, rec)
		defer cleanup()

		id, err := c.DeleteGuardrailConfig(ctx, proj, "g1")
		if err != nil {
			t.Fatalf("DeleteGuardrailConfig: %v", err)
		}
		if rec.method != http.MethodDelete {
			t.Fatalf("method: %s", rec.method)
		}
		if !strings.Contains(rec.path, "projectId="+proj) || !strings.Contains(rec.path, "id=g1") {
			t.Fatalf("path: %s", rec.path)
		}
		if id != "g1" {
			t.Fatalf("deleted id: %q", id)
		}
	})

	t.Run("SecretRefRuleAndValidation", func(t *testing.T) {
		c, _ := NewClient("eg_test", WithBaseURL("http://example.invalid"))

		// List: empty projectID.
		if _, err := c.ListGuardrailConfigs(ctx, ""); err == nil {
			t.Error("want err on empty projectID (List)")
		}
		// Upsert: missing required fields.
		if _, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{}); err == nil {
			t.Error("want err on empty OrgID (Upsert)")
		}
		if _, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{OrgID: org}); err == nil {
			t.Error("want err on empty ProjectID (Upsert)")
		}
		if _, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{OrgID: org, ProjectID: proj}); err == nil {
			t.Error("want err on empty Vendor (Upsert)")
		}
		// Local vendor MUST NOT carry a SecretRef.
		if _, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{
			OrgID: org, ProjectID: proj, Vendor: "data-not-instructions", SecretRef: &secret,
		}); err == nil {
			t.Error("want err: local vendor with SecretRef")
		}
		// Partner vendor MUST carry a SecretRef.
		if _, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{
			OrgID: org, ProjectID: proj, Vendor: "lakera",
		}); err == nil {
			t.Error("want err: partner vendor without SecretRef")
		}
		// VendorChain primary must equal Vendor.
		if _, err := c.UpsertGuardrailConfig(ctx, UpsertGuardrailConfigRequest{
			OrgID: org, ProjectID: proj, Vendor: "lakera", SecretRef: &secret,
			VendorChain: []string{"aporia", "lakera"},
		}); err == nil {
			t.Error("want err: VendorChain[0] != Vendor")
		}
		// Delete: missing required fields.
		if _, err := c.DeleteGuardrailConfig(ctx, "", "g1"); err == nil {
			t.Error("want err on empty projectID (Delete)")
		}
		if _, err := c.DeleteGuardrailConfig(ctx, proj, ""); err == nil {
			t.Error("want err on empty id (Delete)")
		}
	})

	t.Run("IsLocalGuardrailVendor", func(t *testing.T) {
		for _, v := range []string{"local-firewall", "moderated-firewall", "data-not-instructions", "tool-call-circuit-breaker"} {
			if !IsLocalGuardrailVendor(v) {
				t.Errorf("%q should be local", v)
			}
		}
		for _, v := range []string{"lakera", "aporia", "patronus", ""} {
			if IsLocalGuardrailVendor(v) {
				t.Errorf("%q should NOT be local", v)
			}
		}
	})
}
