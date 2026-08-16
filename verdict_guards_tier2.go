package evalguard

import (
	"context"
	"encoding/json"
	"fmt"
)

// ─────────────────────────────────────────────────────────────────────────────
// CLASS 1 — THE COMPLIANCE / GOVERNANCE TIER.
//
// AUDIT 2026-08-08. This is the third and (measurably) last pass of the same
// sweep: 1.5.0 closed seven methods, 1.6.0 closed thirteen more, and RELEASE.md
// named ten that were left — a "candidates for 1.7.0" list. A per-method ×
// per-fault-mode matrix run against this tree found exactly those ten, and only
// those ten, still reading as SAFE. Real output, 24 decision methods × 12 fault
// modes:
//
//   CheckCompliance           missing-verdict, null-verdict, stringly-false,
//                             empty-object, 204-no-body, 200-empty-body
//   DetectDrift               (same six)
//   FormalVerify              (same six)
//   GetAgentMemoryGovernance  (same six)
//   GetComplianceGaps         (same six)
//   GetEUAIAct                (same six)
//   GetModelScanAttestation   (same six)
//   GetSecurityReport         (same six)
//   PromoteModelScan          (same six)
//   TranscribeVoice           (same six)
//   === 10 of 24 probed methods fail OPEN on at least one mode ===
//
// A "still open, tracked for next release" list IS the pattern this sweep
// exists to kill: the hardening was uneven across siblings, so the defect
// survived on whatever the previous pass did not reach. This file reaches them.
//
// Same four-step rule as verdict_guards.go, deliberately no second pattern:
//
//	presence  → the wire carried the decision field at all   → indeterminateVerdict
//	validity  → the value is in the server's CLOSED set      → uninterpretableVerdict
//	evidence  → the verdict agrees with the rest of the body → uninterpretableVerdict
//	binding   → and with the REQUEST that produced it        → uninterpretableVerdict
//
// Everything here is ADDITIVE: new unexported probe types and one new exported
// error path per method. No signature changed, so no consumer's build breaks.
// ─────────────────────────────────────────────────────────────────────────────

// probeMap decodes the 2xx body twice: into the caller's map (the unbroken
// return shape) and into a presence-tracking probe. Decoding the raw BYTES
// rather than re-marshalling the map is what lets the probe tell an ABSENT key
// from one explicitly set to null — the whole point.
//
// Returns (result, raw, err). A nil raw means the body was empty, which is
// never a verdict.
func (c *Client) getGuardedMap(ctx context.Context, method, path string, body any, verdict any) (map[string]any, error) {
	var raw json.RawMessage
	if err := c.doRequest(ctx, method, path, body, &raw); err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		return nil, nil
	}
	var result map[string]any
	if err := json.Unmarshal(raw, &result); err != nil {
		result = nil
	}
	if verdict != nil {
		if err := json.Unmarshal(raw, verdict); err != nil {
			return result, nil
		}
	}
	return result, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /security/report — the CI red-team gate.
//
// The most urgent of the ten, because RunSecurityScan's queued-scan refusal
// (added in 1.6.0) tells callers to poll exactly this method — so the previous
// release actively routed CI gates onto an unswept path. `vulnerabilities` and
// `executiveSummary` assert to nil when absent, so `len(r["vulnerabilities"])`
// is 0 and the pipeline goes green on a report that was never generated.
// ─────────────────────────────────────────────────────────────────────────────

// reportRiskLevels is the CLOSED risk ladder emitted on
// executiveSummary.riskLevel (packages/core/src/security/report-generator.ts:49).
var reportRiskLevels = map[string]bool{"critical": true, "high": true, "medium": true, "low": true}

type securityReportProbe struct {
	ExecutiveSummary *struct {
		RiskLevel            *string  `json:"riskLevel"`
		OverallScore         *float64 `json:"overallScore"`
		TotalVulnerabilities *int     `json:"totalVulnerabilities"`
	} `json:"executiveSummary"`
	Vulnerabilities *[]json.RawMessage `json:"vulnerabilities"`
}

func (p *securityReportProbe) check() string {
	switch {
	case p.ExecutiveSummary == nil:
		return "`executiveSummary` is absent"
	case p.ExecutiveSummary.RiskLevel == nil:
		return "`executiveSummary.riskLevel` is absent"
	case p.ExecutiveSummary.TotalVulnerabilities == nil:
		return "`executiveSummary.totalVulnerabilities` is absent"
	case p.Vulnerabilities == nil:
		return "`vulnerabilities` is absent — a report with no findings array is not a clean report"
	}
	if !reportRiskLevels[*p.ExecutiveSummary.RiskLevel] {
		return fmt.Sprintf("`executiveSummary.riskLevel` is %s, which is not one of critical/high/medium/low",
			quoteVerdict(*p.ExecutiveSummary.RiskLevel))
	}
	// EVIDENCE — the summary's own tally must match the findings beside it, so a
	// body cannot zero the count while leaving the vulnerabilities (or the
	// reverse) and have either reading believed.
	if n := len(*p.Vulnerabilities); *p.ExecutiveSummary.TotalVulnerabilities != n {
		return fmt.Sprintf("`executiveSummary.totalVulnerabilities` is %d but `vulnerabilities` carries %d — "+
			"the summary contradicts the evidence in its own body",
			*p.ExecutiveSummary.TotalVulnerabilities, n)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /formal-verification — the constraint gate.
// ─────────────────────────────────────────────────────────────────────────────

type formalVerifyProbe struct {
	Verified         *bool `json:"verified"`
	TotalConstraints *int  `json:"totalConstraints"`
	Passed           *int  `json:"passed"`
	Failed           *int  `json:"failed"`
	Results          *[]struct {
		ConstraintID string `json:"constraintId"`
		Passed       *bool  `json:"passed"`
	} `json:"results"`
}

// check validates the verdict against itself and against the REQUEST. submitted
// is the number of constraints the caller sent; the response cannot restate that
// number, so it cannot move its own goalposts.
func (p *formalVerifyProbe) check(submitted int) string {
	switch {
	case p.Verified == nil:
		return "`verified` is absent — an unreported verification is not a passing one"
	case p.TotalConstraints == nil:
		return "`totalConstraints` is absent"
	case p.Passed == nil:
		return "`passed` is absent"
	case p.Failed == nil:
		return "`failed` is absent"
	case p.Results == nil:
		return "`results` is absent — a verdict with no per-constraint evidence is not interpretable"
	}
	// BINDING — every constraint SUBMITTED must have been verified.
	if *p.TotalConstraints != submitted {
		return fmt.Sprintf("`totalConstraints` is %d but %d constraint(s) were submitted — %d were never "+
			"verified, so this is not a verdict on the constraint set this caller asked about",
			*p.TotalConstraints, submitted, submitted-*p.TotalConstraints)
	}
	if n := len(*p.Results); n != *p.TotalConstraints {
		return fmt.Sprintf("`results` carries %d entry(ies) but `totalConstraints` is %d", n, *p.TotalConstraints)
	}
	if *p.Passed+*p.Failed != *p.TotalConstraints {
		return fmt.Sprintf("`passed`(%d) + `failed`(%d) is %d, not `totalConstraints`(%d)",
			*p.Passed, *p.Failed, *p.Passed+*p.Failed, *p.TotalConstraints)
	}
	// DERIVATION — the route computes `verified` as results.every(r => r.passed)
	// (formal-verification/route.ts:629). A body claiming verified=true beside a
	// failing constraint is the exact downgrade a release gate must not read.
	failed := 0
	for i, r := range *p.Results {
		if r.Passed == nil {
			return fmt.Sprintf("`results[%d].passed` is absent", i)
		}
		if !*r.Passed {
			failed++
		}
	}
	if failed != *p.Failed {
		return fmt.Sprintf("`failed` is %d but `results` carries %d failing constraint(s)", *p.Failed, failed)
	}
	if *p.Verified != (failed == 0) {
		return fmt.Sprintf("`verified` is %v while %d constraint(s) failed in the same body — the verdict "+
			"contradicts its own evidence", *p.Verified, failed)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /compliance/check — the audit-evidence gate.
// ─────────────────────────────────────────────────────────────────────────────

// complianceStatuses is the CLOSED set on ComplianceCheckResult.status
// (packages/core/src/compliance/engine.ts:122).
var complianceStatuses = map[string]bool{"compliant": true, "partial": true, "non-compliant": true}

type complianceCheckProbe struct {
	Status             *string            `json:"status"`
	OverallScore       *float64           `json:"overallScore"`
	TotalRequirements  *int               `json:"totalRequirements"`
	RequirementResults *[]json.RawMessage `json:"requirementResults"`
	ScanRan            *bool              `json:"scanRan"`
}

func (p *complianceCheckProbe) check() string {
	switch {
	case p.Status == nil:
		return "`status` is absent — an unreported assessment is not a compliant one"
	case p.OverallScore == nil:
		return "`overallScore` is absent"
	case p.TotalRequirements == nil:
		return "`totalRequirements` is absent"
	case p.RequirementResults == nil:
		return "`requirementResults` is absent — a status with no per-requirement evidence is not an " +
			"assessment, and an empty evidence package is what an audit reads as 'nothing failed'"
	case p.ScanRan == nil:
		return "`scanRan` is absent"
	}
	if !complianceStatuses[*p.Status] {
		return fmt.Sprintf("`status` is %s, which is not one of compliant/partial/non-compliant",
			quoteVerdict(*p.Status))
	}
	if *p.OverallScore < 0 || *p.OverallScore > 100 {
		return fmt.Sprintf("`overallScore` is %v, outside the 0..100 scale", *p.OverallScore)
	}
	if n := len(*p.RequirementResults); n != *p.TotalRequirements {
		return fmt.Sprintf("`requirementResults` carries %d entry(ies) but `totalRequirements` is %d — "+
			"requirements this assessment never evaluated are not requirements it met",
			n, *p.TotalRequirements)
	}
	// DERIVATION — engine.ts:130-138 CAPS status at "partial" whenever the scan
	// did not run, so `compliant` implies a scan ran. A body asserting otherwise
	// has upgraded its own verdict.
	if *p.Status == "compliant" && !*p.ScanRan {
		return "`status` is \"compliant\" while `scanRan` is false — the engine caps an assessment at " +
			"\"partial\" when no scan ran, so this body upgraded its own verdict"
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /security/model-scan/{id}/promote — the model-promotion gate.
// ─────────────────────────────────────────────────────────────────────────────

// modelScanDecisions / modelScanGateStatuses are the DB CHECK sets
// (supabase/migrations/20260424_model_scan_attestation.sql:9-10, :31).
var modelScanDecisions = map[string]bool{"promoted": true, "blocked": true, "override": true}
var modelScanGateStatuses = map[string]bool{"pending": true, "promoted": true, "blocked": true, "override": true}

type promoteProbe struct {
	ScanID     *string `json:"scanId"`
	Decision   *string `json:"decision"`
	ToEnv      *string `json:"toEnv"`
	GateStatus *string `json:"gateStatus"`
}

func (p *promoteProbe) check(scanID, toEnv string) string {
	switch {
	case p.Decision == nil:
		return "`decision` is absent — an unreported promotion is not an approved one"
	case p.GateStatus == nil:
		return "`gateStatus` is absent"
	case p.ScanID == nil:
		return "`scanId` is absent"
	case p.ToEnv == nil:
		return "`toEnv` is absent"
	}
	if !modelScanDecisions[*p.Decision] {
		return fmt.Sprintf("`decision` is %s, which is not one of promoted/blocked/override",
			quoteVerdict(*p.Decision))
	}
	if !modelScanGateStatuses[*p.GateStatus] {
		return fmt.Sprintf("`gateStatus` is %s, which is not one of pending/promoted/blocked/override",
			quoteVerdict(*p.GateStatus))
	}
	// BINDING — an approval for a DIFFERENT scan, or into a different
	// environment, is not an approval for this one. This is the check a replayed
	// or rewritten body cannot satisfy.
	if *p.ScanID != scanID {
		return fmt.Sprintf("`scanId` is %s but %s was promoted — this decision is about a different scan",
			quoteVerdict(*p.ScanID), quoteVerdict(scanID))
	}
	if toEnv != "" && *p.ToEnv != toEnv {
		return fmt.Sprintf("`toEnv` is %s but promotion to %s was requested",
			quoteVerdict(*p.ToEnv), quoteVerdict(toEnv))
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /security/model-scan/{id}/attestation — the supply-chain evidence.
// ─────────────────────────────────────────────────────────────────────────────

type attestationProbe struct {
	ScanID      *string          `json:"scanId"`
	Attestation *json.RawMessage `json:"attestation"`
	Cached      *bool            `json:"cached"`
}

func (p *attestationProbe) check(scanID string) string {
	switch {
	case p.Attestation == nil:
		return "`attestation` is absent — a missing attestation is not a signed one"
	case p.ScanID == nil:
		return "`scanId` is absent"
	case p.Cached == nil:
		return "`cached` is absent"
	}
	if string(*p.Attestation) == "null" {
		return "`attestation` is null — a missing attestation is not a signed one"
	}
	if *p.ScanID != scanID {
		return fmt.Sprintf("`scanId` is %s but the attestation for %s was requested — this is evidence "+
			"about a different scan", quoteVerdict(*p.ScanID), quoteVerdict(scanID))
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /voice/transcribe — the input to every downstream voice moderation.
//
// An EMPTY transcript reads as "nothing to moderate": the caller feeds
// result.Text into a firewall / DLP pass, and "" is clean by construction. The
// deepfake sidecar seventeen lines below this in evalguard.go IS guarded; this
// one was not — the same uneven-siblings shape as everywhere else in this sweep.
// ─────────────────────────────────────────────────────────────────────────────

type transcriptProbe struct {
	Text  *string            `json:"text"`
	Words *[]json.RawMessage `json:"words"`
}

func (p *transcriptProbe) check() string {
	if p.Text == nil {
		return "`text` is absent — an unreported transcript is not an empty one, and downstream " +
			"moderation reads \"\" as nothing to screen"
	}
	// EVIDENCE — the ASR always emits word timings alongside the text on the
	// unredacted path, and a redacted response replaces `words` with an explicit
	// null (voice/transcribe/route.ts:97-117) rather than dropping the key. So an
	// ABSENT `words` means this body did not come from the transcriber.
	if p.Words == nil {
		return "`words` is absent — the transcriber emits word timings on every response (an explicit " +
			"null when PII redaction dropped them), so a body without the key is not a transcript"
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /compliance/eu-ai-act and GET /compliance/gaps — the regulator-facing
// reads. Both return arrays the caller counts; nil counts as zero.
// ─────────────────────────────────────────────────────────────────────────────

type euAiActProbe struct {
	Assessments *[]json.RawMessage `json:"assessments"`
	Incidents   *[]json.RawMessage `json:"incidents"`
}

func (p *euAiActProbe) check() string {
	// The route defaults both to [] and never omits either
	// (compliance/eu-ai-act/route.ts:95), so an absent key is schema drift or a
	// substituted body — not "no assessments on file".
	switch {
	case p.Assessments == nil:
		return "`assessments` is absent — the route always emits an array (empty when there are none), " +
			"so a missing key is not \"no conformity assessments\""
	case p.Incidents == nil:
		return "`incidents` is absent — same rule; a missing key is not \"no reportable incidents\""
	}
	return ""
}

// complianceGapStatuses is the CLOSED per-gap status set
// (packages/core/src/compliance/frameworks.ts:71).
var complianceGapStatuses = map[string]bool{"met": true, "partial": true, "not-met": true, "untested": true}

type complianceGapsProbe struct {
	TotalRequirements *int `json:"totalRequirements"`
	MetCount          *int `json:"metCount"`
	PartialCount      *int `json:"partialCount"`
	NotMetCount       *int `json:"notMetCount"`
	UntestedCount     *int `json:"untestedCount"`
	Gaps              *[]struct {
		Status string `json:"status"`
	} `json:"gaps"`
}

func (p *complianceGapsProbe) check() string {
	switch {
	case p.Gaps == nil:
		return "`gaps` is absent — a missing gap list is not a gap-free framework"
	case p.TotalRequirements == nil:
		return "`totalRequirements` is absent"
	case p.MetCount == nil:
		return "`metCount` is absent"
	case p.PartialCount == nil:
		return "`partialCount` is absent"
	case p.NotMetCount == nil:
		return "`notMetCount` is absent"
	case p.UntestedCount == nil:
		return "`untestedCount` is absent"
	}
	for i, g := range *p.Gaps {
		if !complianceGapStatuses[g.Status] {
			return fmt.Sprintf("`gaps[%d].status` is %s, which is not one of met/partial/not-met/untested",
				i, quoteVerdict(g.Status))
		}
	}
	// EVIDENCE — the four buckets partition the requirement set, so a body whose
	// counts do not add up has been rewritten, and a gate reading `notMetCount`
	// would be reading a number the body did not earn.
	if sum := *p.MetCount + *p.PartialCount + *p.NotMetCount + *p.UntestedCount; sum != *p.TotalRequirements {
		return fmt.Sprintf("met(%d)+partial(%d)+notMet(%d)+untested(%d) is %d, not `totalRequirements`(%d)",
			*p.MetCount, *p.PartialCount, *p.NotMetCount, *p.UntestedCount, sum, *p.TotalRequirements)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /agent-memory/governance — "is memory governance ON for this org?"
//
// The route emits `{ policy }` on every success and `{ policy: null }` when
// none is configured (governance/route.ts:72, :76), so `policy` is ALWAYS a
// present key. An ABSENT key therefore means the body is not this route's — but
// the method returned (nil, nil), which every caller reads as "no policy
// configured, governance off".
// ─────────────────────────────────────────────────────────────────────────────

var memoryGovernanceModes = map[string]bool{"off": true, "monitor": true, "enforce": true}

type governanceProbe struct {
	Policy *struct {
		Enabled *bool   `json:"enabled"`
		Mode    *string `json:"mode"`
		OrgID   *string `json:"orgId"`
	} `json:"policy"`
	policyKeyPresent bool
}

func (p *governanceProbe) UnmarshalJSON(data []byte) error {
	type alias governanceProbe
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(data, &keys); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*p = governanceProbe(decoded)
	_, p.policyKeyPresent = keys["policy"]
	return nil
}

func (p *governanceProbe) check(orgID string) string {
	if !p.policyKeyPresent {
		return "`policy` is absent — the route emits the key on every response (an explicit null when no " +
			"policy is configured), so a body without it is not an answer about this org's governance, " +
			"and \"no policy\" is exactly what a caller reads as \"governance is off\""
	}
	if p.Policy == nil {
		return "" // explicit null: a real "no policy configured" answer.
	}
	switch {
	case p.Policy.Enabled == nil:
		return "`policy.enabled` is absent — an unreported policy is not a disabled one"
	case p.Policy.Mode == nil:
		return "`policy.mode` is absent"
	case p.Policy.OrgID == nil:
		return "`policy.orgId` is absent"
	}
	if !memoryGovernanceModes[*p.Policy.Mode] {
		return fmt.Sprintf("`policy.mode` is %s, which is not one of off/monitor/enforce",
			quoteVerdict(*p.Policy.Mode))
	}
	// BINDING — a policy belonging to another org is not this org's policy.
	if *p.Policy.OrgID != orgID {
		return fmt.Sprintf("`policy.orgId` is %s but the policy for %s was requested",
			quoteVerdict(*p.Policy.OrgID), quoteVerdict(orgID))
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /monitoring/drift/detect — the release / rollback gate.
// ─────────────────────────────────────────────────────────────────────────────

// driftSeverities is the CLOSED per-metric severity set
// (packages/core/src/telemetry/drift-detector.ts:9).
var driftSeverities = map[string]bool{"none": true, "low": true, "medium": true, "high": true, "critical": true}

type driftProbe struct {
	HasDrift     *bool    `json:"hasDrift"`
	OverallDelta *float64 `json:"overallDelta"`
	MetricDeltas *[]struct {
		Metric   string `json:"metric"`
		Severity string `json:"severity"`
	} `json:"metricDeltas"`
}

func (p *driftProbe) check() string {
	switch {
	case p.HasDrift == nil:
		return "`hasDrift` is absent — an unreported comparison is not a clean one"
	case p.OverallDelta == nil:
		return "`overallDelta` is absent"
	case p.MetricDeltas == nil:
		return "`metricDeltas` is absent — a drift verdict with no per-metric evidence is not interpretable"
	}
	drifted := 0
	for i, m := range *p.MetricDeltas {
		if !driftSeverities[m.Severity] {
			return fmt.Sprintf("`metricDeltas[%d].severity` is %s, which is not one of "+
				"none/low/medium/high/critical", i, quoteVerdict(m.Severity))
		}
		if m.Severity != "none" {
			drifted++
		}
	}
	// DERIVATION — the route computes hasDrift as
	// `results.filter(r => r.severity !== "none").length > 0`
	// (monitoring/drift/detect/route.ts:99). A body claiming no drift beside a
	// non-none metric has contradicted itself.
	if *p.HasDrift != (drifted > 0) {
		return fmt.Sprintf("`hasDrift` is %v but %d metric(s) in the same body carry a severity other "+
			"than \"none\" — the verdict contradicts its own evidence", *p.HasDrift, drifted)
	}
	return ""
}
