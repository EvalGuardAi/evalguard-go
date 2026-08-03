# Changelog

All notable changes to the EvalGuard Go SDK
(`github.com/EvalGuardAi/evalguard-go`) are documented here. The module is
published by tagging the public mirror repo; `proxy.golang.org` picks up each
tag automatically (see `RELEASE.md`). Keep the `clientVersion` constant in
`evalguard.go` — sent as `x-evalguard-client-version` on every request — in
lockstep with the release tag.

## 1.5.0 — unreleased (prepared 2026-08-03)

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
