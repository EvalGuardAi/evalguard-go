# Changelog

All notable changes to the EvalGuard Go SDK
(`github.com/EvalGuardAi/evalguard-go`) are documented here. The module is
published by tagging the public mirror repo; `proxy.golang.org` picks up each
tag automatically (see `RELEASE.md`). Keep the `clientVersion` constant in
`evalguard.go` — sent as `x-evalguard-client-version` on every request — in
lockstep with the release tag.

## 1.6.0 — 2026-08-07

**Security — `1.5.0` closed seven of the fail-open methods. This closes the
other thirteen.** Same defect, same three outcomes (blocked, allowed, and NO
VERDICT), reached through return types the first pass did not sweep.

`1.5.0` hardened the methods returning a typed struct with a `bool`/`string`
verdict field. The sweep behind this release went through all 158 exported
client methods and found the class was wider in two directions:

- **`map[string]any` returns — the worst shape.** The caller's gate is
  `res["flagged"].(bool)` or `res["action"].(string) == "block"`, and a type
  assertion on an ABSENT key yields the zero value. It does not panic the way a
  WRONG type does, so nothing surfaced. `{}`, an empty body, `null`,
  `{"data":null}`, an envelope with no verdict, and any unrelated HTTP 200 all
  read as "clean image, authentic media, policy allows it, nothing leaked".
- **Structs whose decision lives in a NESTED object or a count**, which the
  first sweep's "does it have a bool/string verdict field" filter walked past.

### Fixed — multimodal moderation

- **`ModerateImage` / `ModerateVideo` / `DetectMediaDeepfake`.** `flagged` and
  `synthetic` absent decode to `false`; `score` and `probability` absent decode
  to `0.0`, which on a 0..1 harm scale is the MOST benign reading available.
  Both natural gates — `if flagged { block }` and `if score > threshold { block }`
  — therefore passed content that was never inspected. `ModerateImage`'s doc
  comment said "Fails closed"; that described the SERVER engine
  (`moderateImage()` returns `flagged:true` when the vision backend throws) and
  was the exact opposite of what the Go client did. The comments now describe
  the client.
- The clip verdicts are re-derived from the per-frame results in the same body
  (flagged = ANY frame, score = MAX frame, `firstFlaggedFrame`, the category
  union, and for deepfake the mean probability), and bound to the REQUEST: how
  many frames were submitted, at what threshold, at what sampling, and which
  media KIND. A response can restate none of that, so it cannot move its own
  goalposts.

### Fixed — runtime enforcement, CI gates and governance

- **`RunGuardrails`** — the most severe of the `map[string]any` cases. It is the
  org-policy twin of `CheckFirewall` on the same request path and had NO
  presence check at all: an absent `action` is `""`, which matches neither
  `"block"` nor `"flag"`, so the text was FORWARDED. The action is now
  re-derived from the reason severities in the same body
  (`critical|high ⇒ block`, `any reason ⇒ flag`, else `allow`), which catches
  the one-word edit that validity against the closed set cannot see.
- **`ScanSecrets` / `ScanIaC` / `CodeScan`** — commit, apply and build gates.
  `findingsCount` absent is `nil` and `findings` absent is `nil`, so
  `if findingsCount > 0 { fail }` passed a scan that opened no file. Counts are
  checked against the findings and the severity tally in the same body, and
  against how many files the caller submitted. (`CodeScan` deliberately excludes
  `info` findings from the tally check — they appear in `findings` but in no
  `severityCounts` bucket, so a naive check would refuse healthy scans.)
- **`LookupVulnerabilities`** — entries are 1:1 with the submitted purls IN
  ORDER, and every summary counter is re-derived from them, so a lookup about
  other packages or about fewer of them cannot pass as a verdict on your
  dependency set.
- **`ClassifyIntent`** — `intent` absent is `""` and `riskScore` absent is `0.0`,
  the bottom of the scale. The classifier returns EARLY with `intent:"harmful"`
  whenever `scores.harmful > 0`, so a body still scoring harm while reporting a
  benign intent is now refused, as is a `sensitivity` below the floor the caller
  asked for (the classifier only ever RAISES it).
- **`IngestRAGDocuments`** — this path runs the same DLP + prompt-injection
  screening `ScanRAGInjection` was hardened for in `1.5.0`, and it was left
  open: `dlp` and `injection` nil read as "no secret, no PII, no injection" for
  documents that were never screened. Both reports are now required, their
  headline counts are checked against the per-document evidence, and a poisoned
  index outside the submitted set is refused — the same rule the RAG scan
  adopted.
- **`AnalyzeShadowAI`** — three `map[string]any` fields, all nil on a 2xx that
  was not an analysis, so every read returned "no PII, no credentials, no risk".
  `calculateRiskScore()` is a fixed additive formula over fields the event
  itself carries, so the score is re-derived term for term — the strongest check
  in the release — and `inputTokens` is bound to `ceil(len(input)/4)` over the
  text this caller actually sent.
- **`ReportAbuse`** — `autoEscalate` and `feedToDetector` are plain bools that
  zero-value to `false`. A CSAM or self-harm report whose triage never arrived
  read as "not escalated, do not feed the detector" and dropped silently out of
  the human review queue. Both flags and the dedup key are re-derived from the
  category and subject the caller filed.

### Fixed — `RunSecurityScan` returned a clean bill of health on EVERY call

This one was not hypothetical, and it was not an edge case.

`DEFAULT_SCAN_DEPTH` is `"full"`, only the 4-strategy `"quick"` set fits
`SYNC_SCAN_STRATEGY_BUDGET`, and `SecurityScanRequest` **carried no `Depth`
field** — so every call this SDK could make resolved to full depth, exceeded the
budget, and was QUEUED. The route answers those with `202
{id, status:"pending", mode:"async", statusUrl}` and **no score, no totalTests,
no severityCounts, no findingsCount**. That decoded to
`Score 0, TotalTests 0, SeverityCounts{0,0,0,0}, FindingsCount 0` with
`err == nil`, so `if res.SeverityCounts.Critical > 0 { fail the build }` passed
100% of the time for a scan that had not started.

- `SecurityScanRequest` gains **`Depth`** (`quick`/`standard`/`full`) and
  **`StrategyIDs`**, so an inline verdict is reachable at all.
- `SecurityScanResult` gains **`Mode`**, **`StatusURL`**, **`ExecutedTests`**,
  **`ErroredTests`**, and **`Queued()`**.
- A queued scan is now REFUSED rather than returned, with the scan id and the
  poll URL in the message so the async flow is still usable. The verdict path
  additionally re-derives `status` from `score >= 70`.

### Added

- `ImageModerationResult`, `VideoModerationResult` / `VideoModerationFrame`,
  `MediaDeepfakeResult` / `MediaDeepfakeFrame` / `DeepfakeLabelScore`,
  `GuardrailsResult` / `GuardrailReason`, `SecretScanResult` /
  `SecretScanFinding`, `IaCScanResult` / `IaCFinding`, `CodeScanResult` /
  `CodeScanFinding`, `SupplyChainLookupResult` / `PurlLookupEntry` /
  `PurlLookupSummary`, `IntentClassification`, `RAGIngestResult` /
  `RAGDlpReport` / `RAGDlpDocumentReport` / `RAGInjectionReport` — each with
  **`HasVerdict()`**, for a caller decoding a stored or proxied body itself.
- `HasVerdict()` on `SecurityScanResult` and `AbuseTriage`.

### Compatibility

**No signature changed.** The seven `map[string]any` methods still return
`map[string]any` — the typed structs are decoded ALONGSIDE the map from the same
bytes and decide the refusal; the map is handed back untouched when the verdict
is real. That is deliberate: `1.5.0` shipped as a MINOR because it changed
BEHAVIOUR and only ADDED public API, and swapping these to typed returns would
break every existing caller's build.

The behaviour change is the same one `1.5.0` made: a 2xx the client cannot
interpret is now `ErrCodeIndeterminate` instead of a readable zero value. Code
already branching on `ErrCodeIndeterminate` needs no change.

### Tests

`media_verdict_test.go` (90 assertions) and `verdict_guards_test.go` cover every
absent-verdict shape against a loopback listener, the explicit-allow verdicts
that must still parse, and the one-field edit that flips a decision while
leaving the evidence behind. Disabling the guards on the shipped surface turns
21 tests and 164 assertions red; every "a real verdict still parses" test stays
green.

Six existing fixtures were bodies the server cannot emit and were corrected:
guardrail reasons with no `severity` (the field `action` is derived FROM), a
secret scan with no `scannedFiles`, an IaC scan with `findingsCount: 2` and no
`findings`, an intent body with a `risk` field the route never had, a triage
with `autoEscalate: true` at severity `"high"`, and a scan with
`status: "completed"` (never emitted) carrying the 0..1 `passRate` where the
route sends `round(passRate * 100)`.

## 1.5.0 — 2026-08-06

**Security — five security decisions could be bypassed by a response that was
not a verdict.** Go's zero value is the same hazard Jackson's primitive default
was in the published Java SDK `1.0.8`: `json.Unmarshal` does **not** error on a
missing key, so `Blocked bool` stayed `false`, `Verdict string` stayed `""`, and
`Clean bool` stayed `false` for any 2xx whose body was not the expected
verdict — schema drift, an API gateway substituting its own envelope on a 2xx,
a truncated payload, `{}`, `{"data":null}`. Measured against a loopback
listener: a body of
`{"success":true,"data":{"score":0.97,"category":"prompt-injection"}}` returned
`err=nil` with `Score=0.97`, `Category="prompt-injection"` and
**`Blocked=false`** — the caller's `if resp.Blocked { deny }` read ALLOW for
content the firewall never evaluated.

There are three outcomes, not two: blocked, allowed, and NO VERDICT. The third
must never collapse into the first.

### Fixed

- **`CheckFirewall` / `CheckFirewallAdvanced` / `CheckFirewallOutputAdvanced`**
  return `ErrCodeIndeterminate` instead of a zero-valued struct when the body
  carried no boolean `blocked`. `CheckFirewallOutputAdvanced` delegates to
  `CheckFirewallAdvanced`, so this also closes model-OUTPUT screening (PII /
  secret-leak / system-prompt-leak).
- **`AuditMcpServer`** — `Verdict` is `"block" | "review" | "pass"`; the zero
  value `""` is none of them, so `if report.Verdict == "block"` DEPLOYED an MCP
  server whose audit never returned.
- **`RunAgentExecRedTeam`** — an absent verdict read as
  `TotalAttacks=0, Breaches=0`: a clean bill of health for a red-team that never
  ran.
- **`ScanRAGInjection`** — `PoisonedIndices` was `nil`, so the natural "keep
  every document not named as poisoned" filter forwarded the whole retrieved set
  to the model.
- **`ScoreVoiceDeepfake`** — on a 0..1 scale, the zero value `0.0` is the MOST
  benign reading, so `if score.Probability > threshold { reject }` accepted every
  sample whose score never arrived.
- **`TestCheckFirewall_EmptyDataField` rewritten.** It asserted "No error,
  `Blocked=false`" for a `{}` body and called that graceful degradation — i.e.
  it PINNED the fail-open as the contract.

**A verdict that is PRESENT but unusable is the same third outcome.** The
refusals above close "no decision field at all". A second sweep found the other
half still open — the field was there, so the presence check passed, but the
value could not be acted on:

- **`ScanRAGInjection` — poison reported that no document could be blamed for.**
  `Clean=false` LOOKS fail-closed (the scan is reporting poison) while the
  consequence is fail-open: with `PoisonedIndices` empty, absent, `null`, or
  naming an index outside the submitted set, the "keep every document not named
  as poisoned" filter documented above names nothing to drop and forwards the
  ENTIRE retrieved set — attack document included. `ScanRAGInjection` now
  returns `ErrCodeIndeterminate` when the result contradicts itself, and
  validates every index against the number of documents submitted.
  `RAGInjectionScanResult.HasVerdict()` reports actionability, not just
  presence.
- **`McpAuditReport.HasVerdict()` / `AgentExecRedTeamResult.HasVerdict()`
  checked PRESENCE where VALIDITY was required.** Both were `Verdict != ""`, so
  any non-empty string passed: a case-drifted `"Block"`, a newer server's
  `"quarantine"`, or a proxy's `"upstream timeout"` all satisfied
  `if report.Verdict == "block" { refuse }` and DEPLOYED a server the audit had
  scored 98/100. Both now match the backend's closed set exactly
  (`block|review|pass`, `breached|attempted|safe`) with no case folding, and an
  unrecognised verdict DENIES — the allowlist rule the Python SDK adopted for
  firewall actions. An old client meeting a newer, stricter verdict must not
  wave it through; if the new verdict is a looser one, the failure is visible
  (a blocked deploy) rather than silent.

Both refusals carry the existing `ErrCodeIndeterminate`, so a caller already
branching on it catches these without a code change.

**A CARDINALITY check is not an IDENTITY check, and a count is not coverage.**
The two refusals above were still checks on PROXIES, and each was bypassed by
mutating ONE field of a body generated by running the shipped backend
(`packages/core/dist/index.js`) into a shape that backend cannot produce:

- **`ScanRAGInjection` compared `poisonedCount` with `len(poisonedIndices)`.**
  `{"poisonedCount":2,"poisonedIndices":[1,1]}` satisfies `2 == 2` while naming
  ONE document, and `[1,4]` satisfies it when the poisoned documents were 1 and
  3 — same count, same range, **different set**. Either way the documented
  filter forwards an attack document. `scanned` was never checked at all, so a
  genuine scan of 3 documents replayed against a 5-document request passed with
  2 documents silently unexamined. The result is now validated as a SET, bound
  to the request: `scanned` must equal the number of documents submitted; the
  indices must be unique, strictly ascending and inside the scanned range; and
  the drop-list must EQUAL the set of documents the report's own `violations`
  incriminate at the requested `minSeverity`. Deleting `violations` does not
  disable that last rule — it makes a non-clean result unevidenced, and refused.
- **`McpAuditReport` — the verdict is DERIVED, so validating the word was not
  enough.** `verdict = summary.critical > 0 ? "block" : summary.high +
  summary.medium > 0 ? "review" : "pass"`. Flipping the one word `"block"` to
  `"pass"` on a report still carrying a critical finding produced a *valid*
  verdict and the deploy gate opened. `riskScore` (which callers gate on) was
  likewise trusted rather than recomputed, and `toolCount` was never compared
  with the number of tools submitted — an audit of 1 of your 3 tools passed. The
  verdict, the summary tallies and the risk score are now all re-derived from
  the findings in the same report, and the tool count is bound to the request.
- **`AgentExecRedTeamResult` — same derived-verdict shape under different
  names.** `verdict = breaches > 0 ? "breached" : dangerousAttempts > 0 ?
  "attempted" : "safe"`, so `"breached"` → `"safe"` on a result still reporting
  a breach turned a real breach into a green build. The verdict is now
  re-derived from the counts, `breaches ⊆ dangerousAttempts ⊆ totalAttacks` is
  enforced, and when the caller supplied the attack prompts the run must have
  executed all of them.

Limit, stated plainly: these rules make a single-field rewrite detectable, not a
wholesale one. A proxy that consistently rewrites three fields (RAG, red-team) or
four (MCP) produces a body indistinguishable from a genuine result for a corpus
where that document was clean. Closing that needs a signed response, not a
client-side rule.

Over-block control: every body a healthy backend can emit still passes,
including a `violations` entry BELOW the requested `minSeverity` (flagged but
correctly not poisoned), the same corpus rescanned at a lower threshold, a
fully-poisoned corpus, a clean corpus, and all three MCP verdicts.

### Added

- `HasVerdict()` on `FirewallCheckResponse`, `McpAuditReport`,
  `AgentExecRedTeamResult`, `RAGInjectionScanResult` and `DeepfakeScore`, and the
  `ErrCodeIndeterminate` (`INDETERMINATE_VERDICT`) error code so callers can
  distinguish "no verdict" from a transport failure.
- Presence-tracking `UnmarshalJSON` on `FirewallCheckResponse`,
  `RAGInjectionScanResult` and `DeepfakeScore`: an explicit `false`/`0` and an
  absent field are indistinguishable by value, so presence is recorded at the
  decode boundary.

### Fixed (build gates)

- `go vet ./...` was RED on the whole package: `retry_backoff_test.go` used
  `t.Context()` (Go 1.24+) while `go.mod` declares `go 1.21`. The failure was
  invisible because a compile error elsewhere in the package masked it. The test
  now uses `context.WithCancel(context.Background())` rather than bumping the
  `go` directive, which would silently drop older toolchains for consumers.

### Compatibility

Source-compatible: no exported type or method was removed. Behaviourally, code
that relied on a non-verdict reading as "not blocked" now gets an error — that
is the point of the release. `MINOR`, not `PATCH`, because `1.4.2` is already
published and reusing a number under changed behaviour is exactly what hid the
published-vs-repo drift in the Java SDK.

`HasVerdict()` on `McpAuditReport`, `AgentExecRedTeamResult` and
`RAGInjectionScanResult` keeps its signature but tightens its meaning from
"a decision field was present" to "a decision this client can act on". Code
that decodes a STORED report and gates on `HasVerdict()` will now see `false`
for a verdict outside the backend's closed set, or for a RAG scan that reports
poison it cannot attribute — both of which previously read as usable. Every
response a healthy backend emits is unaffected: the accepted values are exactly
those `packages/core` produces.

**Note:** `clientVersion` said `1.4.3` while this changelog's newest entry was
headed `1.4.2` — the same version/changelog drift, one step earlier. That entry
has been re-headed `1.4.3 — never published`; see it for why a consumer on the
latest published tag cannot call the methods it documents.

## 1.4.3 — never published

**This entry was published under the heading `## 1.4.2` and that was wrong.**
The `v1.4.2` tag was cut on 2026-07-16 from monorepo commit `b0c3a800e`; the
methods below landed on 2026-07-17 (`63ac81ec2`) and 2026-07-20 (`7ae4edb10`),
*after* the tag, and were bumped to `clientVersion 1.4.3` on 2026-07-22
(`102e86fb5`). No `v1.4.3` tag was ever pushed, so **none of this ever reached
`proxy.golang.org`** — a consumer on the latest published `v1.4.2` calling
`ListGuardrailConfigs` gets a compile error, not a method. Verified 2026-08-03:
the `v1.4.2` module zip on the proxy contains none of `ListScorers`,
`ListGuardrailConfigs`, `UpsertGuardrailConfig`, `DeleteGuardrailConfig`,
`GetAgentMemoryGovernance`, `SetAgentMemoryGovernance`,
`DeleteAgentMemoryGovernance` or `IsLocalGuardrailVendor`.

Everything here ships for the first time in **1.5.0**. Kept as its own heading
rather than folded into 1.5.0 so the gap between "documented" and "downloadable"
stays legible.

Feature-mining reconciliation — adds client methods for the agent-governance
surfaces, mirroring the same agent-governance methods in the sibling SDKs
(current published releases: TS SDK 2.5.4, Python SDK 2.1.5, Java SDK 1.0.7):

- **Gateway guardrail-config** — `ListGuardrailConfigs`, `UpsertGuardrailConfig`,
  and `DeleteGuardrailConfig` manage the per-project inline gateway guardrails
  over `GET`/`POST`/`DELETE /api/v1/gateway/guardrails`. Each row enables one
  vendor: a partner adapter (Lakera / Aporia / Patronus / …, resolved from a
  stored provider key via `SecretRef`) or a local preset that makes no external
  call — `local-firewall`, `moderated-firewall`, and the Wave-2 agent guardrails
  `data-not-instructions` / `tool-call-circuit-breaker`. `IsLocalGuardrailVendor`
  / `LocalGuardrailVendors` and the typed `GuardrailFlagAction`
  (`block`/`redact`/`flag`) model the surface. The SDK enforces the SecretRef
  rule client-side (local vendors must omit it, partner vendors must supply it)
  and requires `VendorChain[0]` to equal the primary `Vendor`, failing fast
  before the request. Upsert is idempotent on `(projectId, vendor)`; upsert and
  delete require Admin role, are audited, and bust the gateway loader cache so a
  change takes effect on the next proxy request.
- **Agent Memory Governance** — `GetAgentMemoryGovernance`,
  `SetAgentMemoryGovernance`, and `DeleteAgentMemoryGovernance` configure the
  org (optionally project-scoped) policy for durable agent-memory writes over
  `GET`/`PUT`/`DELETE /api/v1/agent-memory/governance`. The policy runs in one of
  three modes — `off`, `monitor`, `enforce` (`MemoryGovernanceMode`) — and its
  `MemoryGovernanceConfig` tunes the poisoning-screen threshold
  (`Thresholds.PoisonMinConfidence`), rewrite HITL approval
  (`RequireApprovalOnRewrite`), and provenance requirement (`RequireProvenance`).
  A nil `projectID` targets the org-wide policy; `Get` returns `nil` (no error)
  when none is configured. Every verb requires Admin role and is audited
  server-side.
- **`ListScorers`** — `GET /api/v1/scorers`, returning the typed `Scorer`
  catalogue. Parity with the Java SDK's `listScorers` (`7ae4edb10`).
- **Docs** — README now documents the Agent Memory Governance, Gateway Guardrail
  Config, and Importers surfaces (the ingest endpoints: `CreateTrace`,
  `IngestOTLP`, `IngestShadowAISightings`, `IngestRAGDocuments`).

Also in this unpublished window: bounded retry backoff (`clampRetryDelay` /
`jitteredDelay`, a 10s ceiling and a `Retry-After` fallback) and the internal
`doRaw` split out of `doRequest`.

Backward-compatible; existing methods unchanged. `clientVersion` is `1.4.3`.

## 1.4.2

Published 2026-07-16 from `b0c3a800e` — **the newest version on
`proxy.golang.org` as of 2026-08-03**, and the one every `go get …@latest`
resolves to. Three contract fixes found by live E2E against production:

- **`GenerateAISBOM` was 100% broken (Go-only).** It POSTed `{"projectId": …}`
  to `/ai-sbom/generate`, which validates on `projectName` and 400'd every call.
  Reworked to take a project *name* plus a `GenerateAISBOMOptions` bag
  (`goMod`/`goSum`/`packageJson`/`liveCveScan`/`agents`/…), matching the route
  and the TS/Python SDKs. Signature change; `GetAISBOM` untouched.
- **Base-URL normalization.** A base ending at `…/api` (no `/v1`) 404'd every
  request; `secureBaseURL` now appends `/v1` in that case.
- **`CreateAnnotation.label` enum** (`good`/`bad`/`unsure`) documented,
  `AnnotationLabels` exported, and a client-side check added before the wire.

**On the version jump.** There is no 1.1–1.3 line: no such tags were ever
published. The Go SDK's minor was realigned straight from 1.0.x to the 1.4.x
line so it tracks the EvalGuard core package (`@evalguard/core`, currently
`1.4.7`) rather than carrying an independent minor counter. (An earlier
`clientVersion` of `1.4.0` was the first tag on the realigned line; a
regression test in `evalguard_test.go` guards against the User-Agent version
drifting away from `clientVersion` again, which had happened once at
`evalguard-go/1.2.0` vs `clientVersion 1.4.0`.)

## 1.0.0 – 1.0.4

Initial public releases (as of 2026-04-28, `proxy.golang.org` lists `v1.0.0`
through `v1.0.4`). These predate this changelog file; the Go client tracks the
EvalGuard API surface at parity with the TS/Python/Java SDKs — evals, red-team,
runtime guardrails, tracing, datasets, and the security/compliance endpoints.
No 1.1–1.3 tags follow these — see the note under 1.4.2 for the realignment.
