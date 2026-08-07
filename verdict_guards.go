package evalguard

import (
	"encoding/json"
	"fmt"
	"math"
)

// ─────────────────────────────────────────────────────────────────────────────
// CLASS 1 — THE REST OF THE CLASS.
//
// AUDIT 2026-08-06. 1.5.0 closed seven methods (firewall ×3, MCP audit,
// agent-exec red-team, RAG scan, voice deepfake) and the media sweep in
// evalguard.go closed three more. Sweeping the remaining 148 exported client
// methods for the SAME shape — "a security decision the caller reads out of a
// response, where an absent field reads benign" — turned up ten more, and this
// file closes them.
//
// They divide into two groups by return type, not by severity:
//
//   - map[string]any: RunGuardrails, ScanSecrets, ScanIaC, CodeScan,
//     LookupVulnerabilities, ClassifyIntent, IngestRAGDocuments. A type
//     assertion on an ABSENT key yields the zero value, so `res["action"]` is
//     "" (neither "block" nor "flag"), `res["findingsCount"]` is nil, and
//     `res["findings"]` is nil — "policy allows it, nothing leaked, no
//     misconfiguration".
//   - typed structs: AnalyzeShadowAI, RunSecurityScan, ReportAbuse. Same
//     json.Unmarshal hazard 1.5.0 documented — a missing key is not an error,
//     so `bool` stays false, `string` stays "", `int` stays 0.
//
// Same design as 1.5.0 throughout, deliberately no second pattern:
//
//	presence  → the wire carried the decision field at all   → indeterminateVerdict
//	validity  → the value is in the server's CLOSED set      → uninterpretableVerdict
//	evidence  → the verdict agrees with the rest of the body → uninterpretableVerdict
//	binding   → and with the REQUEST that produced it        → uninterpretableVerdict
//
// The binding step is the one that cannot be forged: a response can restate
// neither how many purls were submitted nor which severity floor was asked for,
// so it cannot move its own goalposts.
//
// Everything here is ADDITIVE to the public API — new types and new HasVerdict
// methods. No signature changes, exactly as 1.5.0 shipped.
// ─────────────────────────────────────────────────────────────────────────────

// verdictFloatTolerance is the slack allowed on a value the server ARRIVED at by
// arithmetic. Counts, comparisons and closed-set matches are exact.
const verdictFloatTolerance = 1e-9

// distinctFiles counts the distinct file paths in a finding set.
func distinctFiles(paths []string) int {
	seen := map[string]bool{}
	for _, p := range paths {
		seen[p] = true
	}
	return len(seen)
}

// tallyMismatch checks a server-emitted per-severity tally against the findings
// in the same body. buckets is the closed set of keys the tally must carry;
// severities outside `counted` are excluded from the comparison (the code-scan
// route, for instance, emits no bucket for "info").
func tallyMismatch(field string, got map[string]int, buckets []string, severities []string, counted map[string]bool) string {
	if got == nil {
		return fmt.Sprintf("`%s` is absent — the body carries findings with no severity tally to justify them", field)
	}
	want := map[string]int{}
	for _, s := range severities {
		if counted == nil || counted[s] {
			want[s]++
		}
	}
	for _, b := range buckets {
		n, ok := got[b]
		if !ok {
			return fmt.Sprintf("`%s` has no %q count — the tally is incomplete", field, b)
		}
		if n != want[b] {
			return fmt.Sprintf("`%s.%s` is %d but `findings` carries %d — the tally does not match the "+
				"evidence in the same body", field, b, n, want[b])
		}
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /guardrails — the runtime allow/flag/block decision.
//
// The most severe entry in this file. RunGuardrails is the org-policy twin of
// CheckFirewall on the same request path, and it was left returning
// map[string]any with no presence check at all: the caller's gate is
// `res["action"].(string) == "block"`, and an absent `action` is "", which is
// neither "block" nor "flag", so the content was FORWARDED.
// ─────────────────────────────────────────────────────────────────────────────

// guardrailActions is the CLOSED set of guardrail decisions
// (packages/core/src/security/firewall.ts: `action: "allow" | "block" | "flag"`).
// Matched EXACTLY — same rule and same reasoning as mcpAuditVerdicts: an action
// this client version does not recognise (a newer, stricter "quarantine") must
// DENY rather than be folded into the nearest one it does know.
var guardrailActions = map[string]bool{"allow": true, "block": true, "flag": true}

// guardrailSeverities is the closed severity set a reason may carry
// (firewall.ts `severity: "critical" | "high" | "medium" | "low"`).
var guardrailSeverities = map[string]bool{"critical": true, "high": true, "medium": true, "low": true}

// GuardrailReason is one rule hit behind a guardrail decision.
type GuardrailReason struct {
	Rule     string `json:"rule"`
	Type     string `json:"type"`
	Detail   string `json:"detail"`
	Severity string `json:"severity"`
}

// GuardrailsResult is the typed view of a POST /guardrails body.
//
// `Action` zero-values to "", which no gate matches, so a 2xx that was not a
// guardrail decision read as "not blocked, not flagged" — an allow.
type GuardrailsResult struct {
	Action    string            `json:"action"`
	Reasons   []GuardrailReason `json:"reasons"`
	LatencyMs float64           `json:"latencyMs"`

	reasonsPresent bool
}

// UnmarshalJSON decodes the decision and records whether `reasons` — the
// EVIDENCE the action is derived from — was on the wire. Requiring it is what
// stops a rewritten body from deleting the evidence to escape the derivation
// check: a body stripped to `{"action":"allow"}` has no verdict, because the
// engine never emits one without reasons.
func (r *GuardrailsResult) UnmarshalJSON(data []byte) error {
	type alias GuardrailsResult
	var probe struct {
		Reasons *[]GuardrailReason `json:"reasons"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = GuardrailsResult(decoded)
	r.reasonsPresent = probe.Reasons != nil
	return nil
}

func (r *GuardrailsResult) missingVerdictField() string {
	switch {
	case r == nil, r.Action == "":
		return "action"
	case !r.reasonsPresent:
		return "reasons"
	}
	return ""
}

// HasVerdict reports whether the body carried a guardrail decision this client
// can ACT ON — one of allow/flag/block, backed by the reasons in the same body.
func (r *GuardrailsResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency() == ""
}

// inconsistency mirrors checkFirewall()'s final derivation exactly:
// `action = hasCritical || hasHigh ? "block" : reasons.length > 0 ? "flag" : "allow"`.
//
// Re-deriving is what catches the one-word edit — "block" to "allow" on a body
// that still carries a critical reason — that validity against the closed set
// cannot see. It is the same edit that turned an MCP "block" into "pass".
func (r *GuardrailsResult) inconsistency() string {
	if r == nil {
		return "nil result"
	}
	if !guardrailActions[r.Action] {
		return fmt.Sprintf("`action` is %s, which is not one of allow/flag/block", quoteVerdict(r.Action))
	}
	if !r.reasonsPresent {
		return "`reasons` is absent — the decision carries no evidence to justify it"
	}
	blocking := false
	for i, reason := range r.Reasons {
		if !guardrailSeverities[reason.Severity] {
			return fmt.Sprintf("`reasons[%d].severity` is %s, outside critical/high/medium/low",
				i, quoteVerdict(reason.Severity))
		}
		if reason.Severity == "critical" || reason.Severity == "high" {
			blocking = true
		}
	}
	want := "allow"
	switch {
	case blocking:
		want = "block"
	case len(r.Reasons) > 0:
		want = "flag"
	}
	if r.Action != want {
		return fmt.Sprintf("`action` is %q but %d reason(s) with the severities in this same body derive "+
			"%q — the policy gate would read a decision the evidence contradicts",
			r.Action, len(r.Reasons), want)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /security/secret-scan — the commit / PR gate.
// ─────────────────────────────────────────────────────────────────────────────

// secretScanSeverities is the closed severity set a finding may carry
// (packages/core/src/secrets/secret-scan.ts SecretFinding.severity).
var secretScanSeverities = []string{"critical", "high", "medium", "low"}

// SecretScanFinding is one redacted secret hit. The raw secret is never on the
// wire — only RedactedMatch.
type SecretScanFinding struct {
	RuleID        string `json:"ruleId"`
	Description   string `json:"description"`
	Severity      string `json:"severity"`
	File          string `json:"file"`
	Line          int    `json:"line"`
	Column        int    `json:"column"`
	RedactedMatch string `json:"redactedMatch"`
	MatchLength   int    `json:"matchLength"`
}

// SecretScanResult is the typed view of a POST /security/secret-scan body.
//
// `FindingsCount` zero-values to 0 and `Findings` to nil, so the natural gate
// `if res.FindingsCount > 0 { fail the commit }` passed a scan that never ran.
type SecretScanResult struct {
	ScannedFiles      int                 `json:"scannedFiles"`
	FilesWithFindings int                 `json:"filesWithFindings"`
	FindingsCount     int                 `json:"findingsCount"`
	Findings          []SecretScanFinding `json:"findings"`
	SeverityCounts    map[string]int      `json:"severityCounts"`

	scannedPresent  bool
	countPresent    bool
	findingsPresent bool
}

func (r *SecretScanResult) UnmarshalJSON(data []byte) error {
	type alias SecretScanResult
	var probe struct {
		ScannedFiles  *int                 `json:"scannedFiles"`
		FindingsCount *int                 `json:"findingsCount"`
		Findings      *[]SecretScanFinding `json:"findings"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = SecretScanResult(decoded)
	r.scannedPresent = probe.ScannedFiles != nil
	r.countPresent = probe.FindingsCount != nil
	r.findingsPresent = probe.Findings != nil
	return nil
}

func (r *SecretScanResult) missingVerdictField() string {
	switch {
	case r == nil, !r.countPresent:
		return "findingsCount"
	case !r.findingsPresent:
		return "findings"
	case !r.scannedPresent:
		return "scannedFiles"
	}
	return ""
}

// HasVerdict reports whether the body carried a scan result this client can ACT
// ON — a finding count backed by the findings and the tally in the same body.
func (r *SecretScanResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(-1) == ""
}

// inconsistency checks the result against itself and against how many files the
// caller submitted. Pass -1 for submitted when that is unknown.
func (r *SecretScanResult) inconsistency(submitted int) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	if r.FindingsCount != len(r.Findings) {
		return fmt.Sprintf("`findingsCount` is %d but `findings` carries %d — a commit gate would be read "+
			"against a number the body did not earn", r.FindingsCount, len(r.Findings))
	}
	// COVERAGE — "0 findings" from a scan that opened no file is not a clean
	// bill of health. The upper bound is what the caller submitted; the scanner
	// may legitimately skip ignored paths, so the check is a bound, not equality.
	if r.ScannedFiles < 1 {
		return "`scannedFiles` is 0 — no file was scanned, so `findingsCount` reports on nothing"
	}
	if submitted >= 0 && r.ScannedFiles > submitted {
		return fmt.Sprintf("`scannedFiles` is %d but only %d file(s) were submitted", r.ScannedFiles, submitted)
	}
	severities := make([]string, 0, len(r.Findings))
	files := make([]string, 0, len(r.Findings))
	for i, f := range r.Findings {
		if !guardrailSeverities[f.Severity] {
			return fmt.Sprintf("`findings[%d].severity` is %s, outside critical/high/medium/low",
				i, quoteVerdict(f.Severity))
		}
		severities = append(severities, f.Severity)
		files = append(files, f.File)
	}
	if msg := tallyMismatch("severityCounts", r.SeverityCounts, secretScanSeverities, severities, nil); msg != "" {
		return msg
	}
	if want := distinctFiles(files); r.FilesWithFindings != want {
		return fmt.Sprintf("`filesWithFindings` is %d but the findings name %d distinct file(s)",
			r.FilesWithFindings, want)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /security/iac-scan — the terraform / k8s apply gate.
// ─────────────────────────────────────────────────────────────────────────────

// IaCFinding is one infrastructure-as-code misconfiguration.
type IaCFinding struct {
	RuleID         string `json:"ruleId"`
	Severity       string `json:"severity"`
	File           string `json:"file"`
	Line           int    `json:"line"`
	Title          string `json:"title"`
	Recommendation string `json:"recommendation"`
}

// IaCScanResult is the typed view of a POST /security/iac-scan body.
type IaCScanResult struct {
	ScannedFiles  int            `json:"scannedFiles"`
	FindingsCount int            `json:"findingsCount"`
	BySeverity    map[string]int `json:"bySeverity"`
	Findings      []IaCFinding   `json:"findings"`

	scannedPresent  bool
	countPresent    bool
	findingsPresent bool
}

func (r *IaCScanResult) UnmarshalJSON(data []byte) error {
	type alias IaCScanResult
	var probe struct {
		ScannedFiles  *int          `json:"scannedFiles"`
		FindingsCount *int          `json:"findingsCount"`
		Findings      *[]IaCFinding `json:"findings"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = IaCScanResult(decoded)
	r.scannedPresent = probe.ScannedFiles != nil
	r.countPresent = probe.FindingsCount != nil
	r.findingsPresent = probe.Findings != nil
	return nil
}

func (r *IaCScanResult) missingVerdictField() string {
	switch {
	case r == nil, !r.countPresent:
		return "findingsCount"
	case !r.findingsPresent:
		return "findings"
	case !r.scannedPresent:
		return "scannedFiles"
	}
	return ""
}

// HasVerdict reports whether the body carried an IaC scan result this client can
// ACT ON.
func (r *IaCScanResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(-1) == ""
}

// inconsistency mirrors scanIacFiles() exactly: `scannedFiles: files.length`
// (unconditional — the IaC scanner skips nothing), `findingsCount` is
// `findings.length`, and `bySeverity` is a straight tally of the findings.
func (r *IaCScanResult) inconsistency(submitted int) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	if r.FindingsCount != len(r.Findings) {
		return fmt.Sprintf("`findingsCount` is %d but `findings` carries %d", r.FindingsCount, len(r.Findings))
	}
	if submitted >= 0 && r.ScannedFiles != submitted {
		return fmt.Sprintf("`scannedFiles` is %d but %d file(s) were submitted — the result does not cover "+
			"the infrastructure this caller asked about", r.ScannedFiles, submitted)
	}
	severities := make([]string, 0, len(r.Findings))
	for i, f := range r.Findings {
		if !guardrailSeverities[f.Severity] {
			return fmt.Sprintf("`findings[%d].severity` is %s, outside critical/high/medium/low",
				i, quoteVerdict(f.Severity))
		}
		severities = append(severities, f.Severity)
	}
	return tallyMismatch("bySeverity", r.BySeverity, secretScanSeverities, severities, nil)
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /security/code-scan — the build gate.
// ─────────────────────────────────────────────────────────────────────────────

// codeScanLanguages is the closed set the route accepts and echoes
// (apps/web/src/app/api/v1/security/code-scan/route.ts).
var codeScanLanguages = map[string]bool{"typescript": true, "javascript": true, "python": true}

// codeScanCounted is the subset of severities the route buckets. `info` is a
// real CodeScanSeverity that appears in `findings` but in NO severityCounts
// bucket (route.ts lines 103-108), so the buckets legitimately sum to less than
// findingsCount — a tally check that did not know this would refuse healthy
// scans.
var codeScanCounted = map[string]bool{"critical": true, "high": true, "medium": true, "low": true}

// codeScanSeverities is the full closed set a finding may carry.
var codeScanSeverities = map[string]bool{
	"critical": true, "high": true, "medium": true, "low": true, "info": true,
}

// CodeScanFinding is one code-security finding.
type CodeScanFinding struct {
	Type           string  `json:"type"`
	Severity       string  `json:"severity"`
	Line           int     `json:"line"`
	Column         int     `json:"column"`
	Code           string  `json:"code"`
	Description    string  `json:"description"`
	Recommendation string  `json:"recommendation"`
	CweID          string  `json:"cweId,omitempty"`
	Source         string  `json:"source,omitempty"`
	Confidence     float64 `json:"confidence,omitempty"`
}

// CodeScanResult is the typed view of a POST /security/code-scan body.
type CodeScanResult struct {
	FilePath       string            `json:"filePath"`
	Language       string            `json:"language"`
	LinesScanned   int               `json:"linesScanned"`
	FindingsCount  int               `json:"findingsCount"`
	Findings       []CodeScanFinding `json:"findings"`
	Semantic       bool              `json:"semantic"`
	SemanticNote   string            `json:"semanticNote,omitempty"`
	SeverityCounts map[string]int    `json:"severityCounts"`

	countPresent    bool
	findingsPresent bool
}

func (r *CodeScanResult) UnmarshalJSON(data []byte) error {
	type alias CodeScanResult
	var probe struct {
		FindingsCount *int               `json:"findingsCount"`
		Findings      *[]CodeScanFinding `json:"findings"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = CodeScanResult(decoded)
	r.countPresent = probe.FindingsCount != nil
	r.findingsPresent = probe.Findings != nil
	return nil
}

func (r *CodeScanResult) missingVerdictField() string {
	switch {
	case r == nil, !r.countPresent:
		return "findingsCount"
	case !r.findingsPresent:
		return "findings"
	case r.Language == "":
		return "language"
	}
	return ""
}

// HasVerdict reports whether the body carried a code-scan result this client can
// ACT ON.
func (r *CodeScanResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency("") == ""
}

// inconsistency checks the result against itself and against the language the
// caller asked for. Pass "" for language when that is unknown.
func (r *CodeScanResult) inconsistency(language string) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	if !codeScanLanguages[r.Language] {
		return fmt.Sprintf("`language` is %s, which is not one of typescript/javascript/python",
			quoteVerdict(r.Language))
	}
	if language != "" && r.Language != language {
		return fmt.Sprintf("`language` is %q but this caller submitted %q — the scan did not parse the "+
			"code as the language it was written in", r.Language, language)
	}
	if r.FindingsCount != len(r.Findings) {
		return fmt.Sprintf("`findingsCount` is %d but `findings` carries %d", r.FindingsCount, len(r.Findings))
	}
	severities := make([]string, 0, len(r.Findings))
	for i, f := range r.Findings {
		if !codeScanSeverities[f.Severity] {
			return fmt.Sprintf("`findings[%d].severity` is %s, outside critical/high/medium/low/info",
				i, quoteVerdict(f.Severity))
		}
		severities = append(severities, f.Severity)
	}
	return tallyMismatch("severityCounts", r.SeverityCounts, secretScanSeverities, severities, codeScanCounted)
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /supply-chain/lookup — the dependency CVE gate.
// ─────────────────────────────────────────────────────────────────────────────

// purlLookupStatuses is the closed per-entry status set
// (packages/core/src/ai-sbom/purl.ts: `status: "ok" | "unsupported" | "invalid"`).
var purlLookupStatuses = map[string]bool{"ok": true, "unsupported": true, "invalid": true}

// PurlLookupEntry is the CVE lookup outcome for one package URL.
type PurlLookupEntry struct {
	Purl            string           `json:"purl"`
	Status          string           `json:"status"`
	Ecosystem       string           `json:"ecosystem,omitempty"`
	Name            string           `json:"name,omitempty"`
	Version         string           `json:"version,omitempty"`
	Vulnerabilities []map[string]any `json:"vulnerabilities,omitempty"`
	Reason          string           `json:"reason,omitempty"`
}

// PurlLookupSummary is the aggregate across every submitted package URL.
type PurlLookupSummary struct {
	Total                int `json:"total"`
	Queried              int `json:"queried"`
	Unsupported          int `json:"unsupported"`
	Invalid              int `json:"invalid"`
	Vulnerable           int `json:"vulnerable"`
	VulnerabilitiesFound int `json:"vulnerabilitiesFound"`
}

// SupplyChainLookupResult is the typed view of a POST /supply-chain/lookup body.
//
// `Entries` nil and `VulnerabilitiesFound` 0 read as "no known CVEs in this
// dependency set" — the answer a release gate wants to hear.
type SupplyChainLookupResult struct {
	Entries                []PurlLookupEntry  `json:"entries"`
	Summary                *PurlLookupSummary `json:"summary"`
	TruncatedAdvisoryCount int                `json:"truncatedAdvisoryCount"`

	entriesPresent bool
}

func (r *SupplyChainLookupResult) UnmarshalJSON(data []byte) error {
	type alias SupplyChainLookupResult
	var probe struct {
		Entries *[]PurlLookupEntry `json:"entries"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = SupplyChainLookupResult(decoded)
	r.entriesPresent = probe.Entries != nil
	return nil
}

func (r *SupplyChainLookupResult) missingVerdictField() string {
	switch {
	case r == nil, !r.entriesPresent:
		return "entries"
	case r.Summary == nil:
		return "summary"
	}
	return ""
}

// HasVerdict reports whether the body carried a lookup result this client can
// ACT ON.
func (r *SupplyChainLookupResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(nil) == ""
}

// inconsistency mirrors lookupVulnerabilitiesByPurl() exactly: entries are 1:1
// with the submitted purls IN ORDER, and every summary field is a count over
// those same entries. Pass nil for purls when they are unknown.
func (r *SupplyChainLookupResult) inconsistency(purls []string) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	// COVERAGE — bound to the REQUEST, entry by entry. A lookup that answered
	// about 2 of the 40 dependencies you submitted, or about different ones, is
	// not a verdict on your dependency set.
	if purls != nil {
		if len(r.Entries) != len(purls) {
			return fmt.Sprintf("`entries` carries %d result(s) but %d package URL(s) were submitted — %d "+
				"dependency(ies) were never looked up", len(r.Entries), len(purls), len(purls)-len(r.Entries))
		}
		for i, p := range purls {
			if r.Entries[i].Purl != p {
				return fmt.Sprintf("`entries[%d].purl` is %s but %s was submitted at that position — the "+
					"results are not about the packages this caller asked about",
					i, quoteVerdict(r.Entries[i].Purl), quoteVerdict(p))
			}
		}
	}
	s := r.Summary
	if s.Total != len(r.Entries) {
		return fmt.Sprintf("`summary.total` is %d but `entries` carries %d", s.Total, len(r.Entries))
	}
	var queried, unsupported, invalid, vulnerable, found int
	for i, e := range r.Entries {
		if !purlLookupStatuses[e.Status] {
			return fmt.Sprintf("`entries[%d].status` is %s, which is not one of ok/unsupported/invalid",
				i, quoteVerdict(e.Status))
		}
		switch e.Status {
		case "ok":
			queried++
			if len(e.Vulnerabilities) > 0 {
				vulnerable++
				found += len(e.Vulnerabilities)
			}
		case "unsupported":
			unsupported++
		case "invalid":
			invalid++
		}
	}
	for _, chk := range []struct {
		field    string
		got, arg int
	}{
		{"queried", s.Queried, queried},
		{"unsupported", s.Unsupported, unsupported},
		{"invalid", s.Invalid, invalid},
		{"vulnerable", s.Vulnerable, vulnerable},
		{"vulnerabilitiesFound", s.VulnerabilitiesFound, found},
	} {
		if chk.got != chk.arg {
			return fmt.Sprintf("`summary.%s` is %d but the entries in this same body carry %d — a release "+
				"gate would be read against a number the body did not earn", chk.field, chk.got, chk.arg)
		}
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /governance/intent/classify — the data-sensitivity / governance gate.
// ─────────────────────────────────────────────────────────────────────────────

// promptIntents is the CLOSED 13-value intent set
// (packages/core/src/intent/index.ts `PromptIntent`).
var promptIntents = map[string]bool{
	"code-generation": true, "data-analysis": true, "research": true, "content-creation": true,
	"financial": true, "legal": true, "hr-personnel": true, "security-ops": true,
	"customer-support": true, "translation": true, "summarization": true, "harmful": true,
	"general": true,
}

// sensitivityRank is the ordered sensitivity ladder (intent/index.ts
// SENSITIVITY_RANK). Rank is what lets the client check the server honoured the
// floor it was asked for — a downgrade below the floor is the fail-open.
var sensitivityRank = map[string]int{"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

// IntentClassification is the typed view of a POST /governance/intent/classify
// body.
//
// `Intent` zero-values to "" and `RiskScore` to 0.0 — the bottom of the risk
// scale — so a governance gate reading `if res.RiskScore > 0.5 { escalate }`
// waved through every prompt whose classification never arrived.
type IntentClassification struct {
	Intent      string             `json:"intent"`
	Confidence  float64            `json:"confidence"`
	Sensitivity string             `json:"sensitivity"`
	RiskScore   float64            `json:"riskScore"`
	Signals     []string           `json:"signals"`
	Scores      map[string]float64 `json:"scores"`

	confidencePresent bool
	riskPresent       bool
}

func (r *IntentClassification) UnmarshalJSON(data []byte) error {
	type alias IntentClassification
	var probe struct {
		Confidence *float64 `json:"confidence"`
		RiskScore  *float64 `json:"riskScore"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = IntentClassification(decoded)
	r.confidencePresent = probe.Confidence != nil
	r.riskPresent = probe.RiskScore != nil
	return nil
}

func (r *IntentClassification) missingVerdictField() string {
	switch {
	case r == nil, r.Intent == "":
		return "intent"
	case r.Sensitivity == "":
		return "sensitivity"
	case !r.riskPresent:
		return "riskScore"
	case !r.confidencePresent:
		return "confidence"
	case r.Scores == nil:
		return "scores"
	}
	return ""
}

// HasVerdict reports whether the body carried a classification this client can
// ACT ON.
func (r *IntentClassification) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency("") == ""
}

// inconsistency checks the classification against itself, against the full score
// table in the same body, and against the sensitivity floor the caller asked
// for. Pass "" for floor when there was none.
func (r *IntentClassification) inconsistency(floor string) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	if !promptIntents[r.Intent] {
		return fmt.Sprintf("`intent` is %s, which is not one of the 13 intents this client knows",
			quoteVerdict(r.Intent))
	}
	rank, ok := sensitivityRank[r.Sensitivity]
	if !ok {
		return fmt.Sprintf("`sensitivity` is %s, which is not one of public/internal/confidential/restricted",
			quoteVerdict(r.Sensitivity))
	}
	if r.RiskScore < 0 || r.RiskScore > 1 {
		return fmt.Sprintf("`riskScore` is %v, outside the 0..1 scale", r.RiskScore)
	}
	if r.Confidence < 0 || r.Confidence > 1 {
		return fmt.Sprintf("`confidence` is %v, outside the 0..1 scale", r.Confidence)
	}
	// EVIDENCE — the classifier always emits the full 13-entry score table, so
	// requiring it stops a rewritten body from deleting the table to escape the
	// two derivation checks below.
	for intent := range promptIntents {
		if _, ok := r.Scores[intent]; !ok {
			return fmt.Sprintf("`scores` has no %q entry — the score table is incomplete, so `intent` "+
				"cannot be checked against the evidence", intent)
		}
	}
	// DERIVATION 1 — classifyPromptIntent() returns EARLY with
	// `intent: "harmful"` whenever `scores.harmful > 0`, overriding the argmax.
	// So a body scoring harm while reporting any other intent is the exact
	// downgrade a governance gate must not read.
	if r.Scores["harmful"] > 0 && r.Intent != "harmful" {
		return fmt.Sprintf("`scores.harmful` is %v but `intent` is %q — the classifier reports \"harmful\" "+
			"whenever the harm score is non-zero, so this body downgrades its own finding",
			r.Scores["harmful"], r.Intent)
	}
	// DERIVATION 2 — outside the harmful override the intent is the ARGMAX of
	// the table. The tie-break order is not re-implemented (it would reject a
	// future reordering for no safety gain); being A maximum is what matters.
	if r.Intent != "harmful" {
		for intent, score := range r.Scores {
			if score > r.Scores[r.Intent] {
				return fmt.Sprintf("`intent` is %q scoring %v but %q scores %v — the reported intent is "+
					"not the top of the table in its own body",
					r.Intent, r.Scores[r.Intent], intent, score)
			}
		}
	}
	// BINDING — sensitivity is seeded at the caller's floor and only ever RAISED
	// (maxSensitivity), so anything below the floor this caller asked for is a
	// downgrade the server cannot legitimately have produced.
	if floor != "" {
		want, ok := sensitivityRank[floor]
		if ok && rank < want {
			return fmt.Sprintf("`sensitivity` is %q but this caller asked for a floor of %q — the "+
				"classifier only ever raises sensitivity above the floor", r.Sensitivity, floor)
		}
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /rag/ingest — DLP + prompt-injection screening on every ingested chunk.
// ─────────────────────────────────────────────────────────────────────────────

// ragDlpModes / ragInjectionModes are the closed mode sets
// (packages/core/src/ingestion/dlp-screen.ts DlpMode,
// packages/core/src/gateway/rag-injector.ts).
var ragDlpModes = map[string]bool{"off": true, "scan": true, "redact": true, "block": true}
var ragInjectionModes = map[string]bool{"off": true, "scan": true, "block": true}

// RAGDlpDocumentReport is the per-document DLP report.
type RAGDlpDocumentReport struct {
	DocumentIndex  int              `json:"documentIndex"`
	DocumentID     string           `json:"documentId,omitempty"`
	Secrets        []map[string]any `json:"secrets"`
	PIIEntityTypes []string         `json:"piiEntityTypes"`
	PIICount       int              `json:"piiCount"`
}

// RAGDlpReport is the DLP half of a POST /rag/ingest body.
type RAGDlpReport struct {
	Mode         string                 `json:"mode"`
	SecretsFound int                    `json:"secretsFound"`
	PIIFound     int                    `json:"piiFound"`
	Reports      []RAGDlpDocumentReport `json:"reports"`
}

// RAGInjectionReport is the prompt-injection half of a POST /rag/ingest body.
type RAGInjectionReport struct {
	Mode            string           `json:"mode"`
	PoisonedCount   int              `json:"poisonedCount"`
	PoisonedIndices []int            `json:"poisonedIndices"`
	Flagged         []map[string]any `json:"flagged"`
}

// RAGIngestResult is the typed view of a POST /rag/ingest body.
//
// The ingest path runs the SAME screening ScanRAGInjection was hardened for in
// 1.5.0, and it was left open: `Dlp` and `Injection` nil read as "no secret, no
// PII, no injection" for documents that were never screened.
type RAGIngestResult struct {
	Chunks     []map[string]any    `json:"chunks"`
	ChunkCount int                 `json:"chunkCount"`
	Embedded   bool                `json:"embedded"`
	Model      string              `json:"model,omitempty"`
	Dlp        *RAGDlpReport       `json:"dlp"`
	Injection  *RAGInjectionReport `json:"injection"`

	chunksPresent bool
	countPresent  bool
}

func (r *RAGIngestResult) UnmarshalJSON(data []byte) error {
	type alias RAGIngestResult
	var probe struct {
		Chunks     *[]map[string]any `json:"chunks"`
		ChunkCount *int              `json:"chunkCount"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = RAGIngestResult(decoded)
	r.chunksPresent = probe.Chunks != nil
	r.countPresent = probe.ChunkCount != nil
	return nil
}

// missingVerdictField requires BOTH screening reports.
//
// They are conditional on the wire — the route omits each when its mode is
// "off" — but IngestRAGRequest exposes no mode knob, so every request this SDK
// sends takes the server default of "scan" and both reports are always emitted.
// An absent report therefore means the screening did not run, which is exactly
// the case that must not read as clean.
func (r *RAGIngestResult) missingVerdictField() string {
	switch {
	case r == nil, !r.chunksPresent:
		return "chunks"
	case !r.countPresent:
		return "chunkCount"
	case r.Dlp == nil:
		return "dlp"
	case r.Injection == nil:
		return "injection"
	}
	return ""
}

// HasVerdict reports whether the body carried both screening reports, backed by
// the per-document evidence in the same body.
func (r *RAGIngestResult) HasVerdict() bool {
	return r != nil && r.missingVerdictField() == "" && r.inconsistency(-1) == ""
}

// inconsistency checks both reports against themselves and against how many
// documents the caller submitted. Pass -1 for docCount when unknown.
func (r *RAGIngestResult) inconsistency(docCount int) string {
	if r == nil {
		return "nil result"
	}
	if f := r.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent — the ingest pipeline's screening did not report", f)
	}
	if r.ChunkCount != len(r.Chunks) {
		return fmt.Sprintf("`chunkCount` is %d but `chunks` carries %d", r.ChunkCount, len(r.Chunks))
	}
	if !ragDlpModes[r.Dlp.Mode] {
		return fmt.Sprintf("`dlp.mode` is %s, which is not one of off/scan/redact/block", quoteVerdict(r.Dlp.Mode))
	}
	if !ragInjectionModes[r.Injection.Mode] {
		return fmt.Sprintf("`injection.mode` is %s, which is not one of off/scan/block",
			quoteVerdict(r.Injection.Mode))
	}
	// A mode of "off" on a request that could not ask for it means something
	// between the pipeline and this client turned the screening off.
	if r.Dlp.Mode == "off" || r.Injection.Mode == "off" {
		return fmt.Sprintf("screening reported mode dlp=%q injection=%q, but this client cannot request "+
			"\"off\" — the documents were not screened", r.Dlp.Mode, r.Injection.Mode)
	}
	// EVIDENCE — the two headline counts are sums over the per-document reports
	// in the same body, so a rewritten body cannot zero them and keep the
	// evidence.
	secrets, pii := 0, 0
	for _, rep := range r.Dlp.Reports {
		secrets += len(rep.Secrets)
		pii += rep.PIICount
	}
	if r.Dlp.SecretsFound != secrets {
		return fmt.Sprintf("`dlp.secretsFound` is %d but the per-document reports carry %d",
			r.Dlp.SecretsFound, secrets)
	}
	if r.Dlp.PIIFound != pii {
		return fmt.Sprintf("`dlp.piiFound` is %d but the per-document reports carry %d", r.Dlp.PIIFound, pii)
	}
	if r.Injection.PoisonedCount != len(r.Injection.PoisonedIndices) {
		return fmt.Sprintf("`injection.poisonedCount` is %d but `poisonedIndices` names %d document(s) — "+
			"the natural \"drop every document named as poisoned\" filter would drop the wrong number",
			r.Injection.PoisonedCount, len(r.Injection.PoisonedIndices))
	}
	// BINDING — an index the caller cannot map back to a submitted document is
	// as useless as no index at all. Same rule ScanRAGInjection adopted.
	if docCount >= 0 {
		for _, idx := range r.Injection.PoisonedIndices {
			if idx < 0 || idx >= docCount {
				return fmt.Sprintf("`injection.poisonedIndices` names document %d but only %d were "+
					"submitted — the caller cannot act on it", idx, docCount)
			}
		}
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /shadow-ai — unauthorized-model / PII / credential egress detection.
// ─────────────────────────────────────────────────────────────────────────────

// shadowAIActions is the CLOSED action set
// (packages/core/src/gateway/shadow-ai-detector.ts:
// `action: "allowed" | "blocked" | "flagged"`). Note this is a DIFFERENT set
// from the guardrails route's allow/block/flag — matching either one against the
// other is how a gate silently stops matching at all.
var shadowAIActions = map[string]bool{"allowed": true, "blocked": true, "flagged": true}

// shadowAIEvent is the private typed view of ShadowAIResult.Event. ShadowAIResult
// keeps its three map[string]any fields — changing them would break every
// caller's compile — so the typed view is decoded alongside them and is what the
// refusal is decided on.
type shadowAIEvent struct {
	Timestamp             float64  `json:"timestamp"`
	UserID                string   `json:"userId"`
	Provider              string   `json:"provider"`
	Model                 string   `json:"model"`
	Authorized            bool     `json:"authorized"`
	PIIDetected           bool     `json:"piiDetected"`
	PIITypes              []string `json:"piiTypes"`
	SensitiveDataDetected bool     `json:"sensitiveDataDetected"`
	SensitiveDataTypes    []string `json:"sensitiveDataTypes"`
	InputTokens           int      `json:"inputTokens"`
	Action                string   `json:"action"`
	RiskScore             float64  `json:"riskScore"`

	riskPresent bool
}

func (e *shadowAIEvent) UnmarshalJSON(data []byte) error {
	type alias shadowAIEvent
	var probe struct {
		RiskScore *float64 `json:"riskScore"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*e = shadowAIEvent(decoded)
	e.riskPresent = probe.RiskScore != nil
	return nil
}

// shadowAIDetection is the private typed view of the piiDetails /
// sensitiveDataDetails sub-objects.
type shadowAIDetection struct {
	Detected bool     `json:"detected"`
	Types    []string `json:"types"`

	detectedPresent bool
}

func (d *shadowAIDetection) UnmarshalJSON(data []byte) error {
	type alias shadowAIDetection
	var probe struct {
		Detected *bool `json:"detected"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return err
	}
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*d = shadowAIDetection(decoded)
	d.detectedPresent = probe.Detected != nil
	return nil
}

// shadowAIVerdict is the typed shadow of the whole POST /shadow-ai body.
type shadowAIVerdict struct {
	Event      *shadowAIEvent     `json:"event"`
	PII        *shadowAIDetection `json:"piiDetails"`
	Sensitive  *shadowAIDetection `json:"sensitiveDataDetails"`
	inputChars int
}

func (v *shadowAIVerdict) missingVerdictField() string {
	switch {
	case v == nil, v.Event == nil:
		return "event"
	case v.Event.Action == "":
		return "event.action"
	case !v.Event.riskPresent:
		return "event.riskScore"
	case v.PII == nil, !v.PII.detectedPresent:
		return "piiDetails.detected"
	case v.Sensitive == nil, !v.Sensitive.detectedPresent:
		return "sensitiveDataDetails.detected"
	}
	return ""
}

// inconsistency re-derives riskScore and action from the event's own fields.
//
// calculateRiskScore() (shadow-ai-detector.ts) is a fixed additive formula over
// exactly the fields the event carries, so the score is fully re-derivable — the
// strongest check available anywhere in this file. `inputTokens` is
// `Math.ceil(input.length / 4)`, which binds the whole event to the REQUEST.
func (v *shadowAIVerdict) inconsistency() string {
	if f := v.missingVerdictField(); f != "" {
		return fmt.Sprintf("`%s` is absent", f)
	}
	e := v.Event
	if !shadowAIActions[e.Action] {
		return fmt.Sprintf("`event.action` is %s, which is not one of allowed/flagged/blocked",
			quoteVerdict(e.Action))
	}
	if e.RiskScore < 0 || e.RiskScore > 100 {
		return fmt.Sprintf("`event.riskScore` is %v, outside the 0..100 scale", e.RiskScore)
	}
	// The two detail objects restate the event's own booleans; a body where they
	// disagree has been edited on one side only.
	if e.PIIDetected != v.PII.Detected {
		return fmt.Sprintf("`event.piiDetected` is %t but `piiDetails.detected` is %t",
			e.PIIDetected, v.PII.Detected)
	}
	if e.SensitiveDataDetected != v.Sensitive.Detected {
		return fmt.Sprintf("`event.sensitiveDataDetected` is %t but `sensitiveDataDetails.detected` is %t",
			e.SensitiveDataDetected, v.Sensitive.Detected)
	}
	if v.PII.Detected != (len(v.PII.Types) > 0) {
		return fmt.Sprintf("`piiDetails.detected` is %t but %d PII type(s) are named",
			v.PII.Detected, len(v.PII.Types))
	}
	if v.Sensitive.Detected != (len(v.Sensitive.Types) > 0) {
		return fmt.Sprintf("`sensitiveDataDetails.detected` is %t but %d sensitive type(s) are named",
			v.Sensitive.Detected, len(v.Sensitive.Types))
	}
	// BINDING — inputTokens is ceil(len(input)/4) over the text THIS caller sent.
	if v.inputChars > 0 {
		want := (v.inputChars + 3) / 4
		if e.InputTokens != want {
			return fmt.Sprintf("`event.inputTokens` is %d but the %d-character input this caller submitted "+
				"is %d token(s) — the event is not about this request", e.InputTokens, v.inputChars, want)
		}
	}
	// DERIVATION — calculateRiskScore(), term for term.
	score := 0
	if !e.Authorized {
		score += 30
	}
	if e.PIIDetected {
		score += 25
	}
	if e.SensitiveDataDetected {
		score += 25
	}
	if len(e.PIITypes) > 2 {
		score += 10
	}
	for _, t := range e.SensitiveDataTypes {
		if t == "privateKey" || t == "awsKey" {
			score += 10
		}
	}
	if e.InputTokens > 10000 {
		score += 5
	}
	if score > 100 {
		score = 100
	}
	if math.Abs(e.RiskScore-float64(score)) > verdictFloatTolerance {
		return fmt.Sprintf("`event.riskScore` is %v but the signals in this same event weigh %d — an "+
			"escalation threshold would be read against a number the event did not earn", e.RiskScore, score)
	}
	// DERIVATION — analyzeRequest() sets "blocked" only for an unauthorized
	// model, else "flagged" at riskScore >= 50. So a score at or above 50 can
	// never be "allowed", whatever the policy.
	if score >= 50 && e.Action == "allowed" {
		return fmt.Sprintf("`event.action` is \"allowed\" at riskScore %v — the detector flags at 50 and "+
			"above, so these cannot both be true", e.RiskScore)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /security — the synchronous red-team scan.
// ─────────────────────────────────────────────────────────────────────────────

// securityScanStatuses is the CLOSED status set a SUCCESS body can carry.
//
// Two response shapes, not one. The route runs the scan inline and answers 201
// with `status: "passed" | "failed"` ONLY when the resolved strategy set fits
// SYNC_SCAN_STRATEGY_BUDGET (4, the curated `quick` set). Otherwise it queues
// the scan and answers 202 with `status: "pending"`, `mode: "async"` and a
// statusUrl — and NO score, totalTests, severityCounts or findingsCount at all.
// "error"/"running" are DB states the route never puts in a success body: a scan
// that executed nothing becomes a 502 SCAN_TARGET_UNREACHABLE instead.
var securityScanStatuses = map[string]bool{"passed": true, "failed": true, "pending": true}

// HasVerdict reports whether the scan returned a result this client can ACT ON —
// a scan that RAN, with counts that agree.
//
// AUDIT 2026-08-06: SecurityScanResult is the structural sibling of
// AgentExecRedTeamResult, which 1.5.0 hardened, and it was missed. Every numeric
// field zero-values to 0 and Status to "", so a 2xx that was not a scan read as
// "0 tests, 0 findings, 0 critical" — a clean red-team that never ran.
//
// It is FALSE for a queued scan on purpose. DEFAULT_SCAN_DEPTH is "full", and
// SecurityScanRequest sent no depth until this release, so every Go call landed
// on the 202 path and got exactly that all-zero summary back with err == nil:
// `if res.SeverityCounts.Critical > 0 { fail the build }` passed 100% of the
// time, for a scan that had not started. That was not an edge case — it was the
// only path the Go SDK could reach.
func (r *SecurityScanResult) HasVerdict() bool {
	return r != nil && r.Status != "" && r.Status != "pending" && r.inconsistency() == ""
}

// Queued reports whether this is the 202 acknowledgement of a scan handed to the
// worker rather than a finished verdict. Poll StatusURL, or set
// SecurityScanRequest.Depth to "quick" to run the scan inline.
func (r *SecurityScanResult) Queued() bool {
	return r != nil && r.Status == "pending"
}

// inconsistency mirrors the route's derivations: `score = round(passRate * 100)`
// and `status = score >= 70 ? "passed" : "failed"`, with `totalTests` and
// `findingsCount` both equal to `findings.length`.
func (r *SecurityScanResult) inconsistency() string {
	if r == nil {
		return "nil result"
	}
	if !securityScanStatuses[r.Status] {
		return fmt.Sprintf("`status` is %s, which is not one of passed/failed/pending",
			quoteVerdict(r.Status))
	}
	// A queued scan carries no counts to check — and must never be read as one.
	// The caller-facing refusal is raised by RunSecurityScan; here it is simply
	// not an inconsistency.
	if r.Status == "pending" {
		if r.ID == "" {
			return "`status` is \"pending\" but no scan `id` was returned, so the scan cannot be polled"
		}
		return ""
	}
	// COVERAGE — the twin of RunAgentExecRedTeam's `totalAttacks` check: a scan
	// that ran no test is a clean bill of health for a run that never happened.
	if r.TotalTests < 1 {
		return "`totalTests` is 0 — no test was run, so `status` reports on nothing"
	}
	if r.Score < 0 || r.Score > 100 {
		return fmt.Sprintf("`score` is %v, outside the 0..100 scale", r.Score)
	}
	// The route emits totalTests and findingsCount from the SAME findings array.
	if r.FindingsCount != r.TotalTests {
		return fmt.Sprintf("`findingsCount` is %d but `totalTests` is %d — the route emits both from the "+
			"same findings array", r.FindingsCount, r.TotalTests)
	}
	sc := r.SeverityCounts
	if sc.Critical < 0 || sc.High < 0 || sc.Medium < 0 || sc.Low < 0 {
		return fmt.Sprintf("negative severity counts (%+v)", sc)
	}
	// Severity counters cover only findings that neither passed nor errored, so
	// they are a SUBSET of the findings — never more.
	if sum := sc.Critical + sc.High + sc.Medium + sc.Low; sum > r.FindingsCount {
		return fmt.Sprintf("`severityCounts` totals %d but `findingsCount` is %d — the tally counts more "+
			"vulnerabilities than the scan produced findings", sum, r.FindingsCount)
	}
	// DERIVATION — the one-word edit that turns "failed" into "passed" on a scan
	// that scored 12 is exactly what a build gate reading `status == "passed"`
	// must not accept.
	want := "failed"
	if r.Score >= 70 {
		want = "passed"
	}
	if r.Status != want {
		return fmt.Sprintf("`status` is %q but a score of %v derives %q — the route sets passed at 70 and "+
			"above", r.Status, r.Score, want)
	}
	return ""
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /abuse-reports — the trust-and-safety auto-triage verdict.
// ─────────────────────────────────────────────────────────────────────────────

// abuseSeverities / abuseCategories are the closed sets
// (packages/core/src/security/abuse-reports.ts).
var abuseSeverities = map[string]bool{"low": true, "medium": true, "high": true, "critical": true}
var abuseCategories = map[string]bool{
	"csam": true, "violence": true, "self_harm": true, "harassment": true, "hate": true,
	"fraud": true, "privacy": true, "spam": true, "other": true,
}

// HasVerdict reports whether the triage returned a verdict this client can ACT
// ON — a recognised severity whose escalation flags follow from it.
//
// AUDIT 2026-08-06: `AutoEscalate` and `FeedToDetector` are plain bools that
// zero-value to false, and `Severity` to "". A CSAM or self-harm report whose
// triage never arrived therefore read as "not escalated, do not feed the
// detector" and dropped silently out of the human review queue.
func (t *AbuseTriage) HasVerdict() bool {
	return t != nil && t.Severity != "" && t.inconsistency("", "") == ""
}

// inconsistency mirrors triageAbuseReport() exactly:
// `autoEscalate = severity === "critical"` and
// `feedToDetector = !!subjectId && (severity === "high" || severity === "critical")`,
// with `dedupKey = ${category}:${subjectId ?? "anon"}`. Pass "" for category /
// subjectID when the request is unknown.
func (t *AbuseTriage) inconsistency(category, subjectID string) string {
	if t == nil {
		return "nil triage"
	}
	if !abuseSeverities[t.Severity] {
		return fmt.Sprintf("`severity` is %s, which is not one of low/medium/high/critical",
			quoteVerdict(t.Severity))
	}
	if !abuseCategories[t.Category] {
		return fmt.Sprintf("`category` is %s, which is not one of the nine abuse categories",
			quoteVerdict(t.Category))
	}
	// BINDING — the triage echoes the caller's category, so a triage about a
	// different category is a triage about a different report.
	if category != "" && t.Category != category {
		return fmt.Sprintf("`category` is %q but this caller reported %q", t.Category, category)
	}
	if len(t.Reasons) == 0 {
		return "`reasons` is empty — the triage always states at least the category reason, so this " +
			"verdict carries no evidence"
	}
	// DERIVATION — both escalation flags follow from the POST-escalation
	// severity, so flipping either one alone is detectable.
	if want := t.Severity == "critical"; t.AutoEscalate != want {
		return fmt.Sprintf("`autoEscalate` is %t but severity %q derives %t — auto-escalation is exactly "+
			"the critical tier", t.AutoEscalate, t.Severity, want)
	}
	if category != "" {
		want := subjectID != "" && (t.Severity == "high" || t.Severity == "critical")
		if t.FeedToDetector != want {
			return fmt.Sprintf("`feedToDetector` is %t but a %q report %s a subject derives %t",
				t.FeedToDetector, t.Severity,
				map[bool]string{true: "with", false: "without"}[subjectID != ""], want)
		}
		subject := subjectID
		if subject == "" {
			subject = "anon"
		}
		if want := category + ":" + subject; t.DedupKey != want {
			return fmt.Sprintf("`dedupKey` is %s but this report derives %s — the triage is keyed to a "+
				"different subject", quoteVerdict(t.DedupKey), quoteVerdict(want))
		}
	}
	return ""
}
