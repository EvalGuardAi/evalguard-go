# EvalGuard Go SDK

[![Go Reference](https://pkg.go.dev/badge/github.com/EvalGuardAi/evalguard-go.svg)](https://pkg.go.dev/github.com/EvalGuardAi/evalguard-go)

Go client for [EvalGuard](https://evalguard.ai) — LLM evaluation, red-team
testing, and runtime guardrails.

## Install

```bash
go get github.com/EvalGuardAi/evalguard-go@latest
```

This module is mirrored to the public `EvalGuardAi/evalguard-go` repo (the
internal monorepo where the SDK source lives is private). Tags on that
public mirror cut releases via `proxy.golang.org` — see
`.github/workflows/publish-go-sdk.yml` and `RELEASE.md`.

## Quick start

```go
package main

import (
    "context"
    "log"
    "time"

    evalguard "github.com/EvalGuardAi/evalguard-go"
)

func main() {
    client, err := evalguard.NewClient("your-api-key",
        evalguard.WithBaseURL("https://evalguard.ai/api/v1"),
        evalguard.WithTimeout(30*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    // RunEval starts an async run; poll GetEval(started.ID) for results.
    started, err := client.RunEval(context.Background(), &evalguard.RunEvalRequest{
        Name:      "regression-suite",
        ProjectID: "proj_abc123",
        Model:     "gpt-4o",
        Prompt:    "Answer concisely: {{input}}",
        Cases:     []evalguard.EvalCase{{Input: "2+2?", ExpectedOutput: "4"}},
        Scorers:   []string{"exact-match"},
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("started run %s (status=%s, %d tests)", started.ID, started.Status, started.TotalTests)
}
```

## Agent Memory Governance

Configure the org-level (optionally project-scoped) policy that governs durable
agent-memory writes — poisoning-screen thresholds, human-in-the-loop approval on
rewrites, and provenance requirements. Every verb requires **Admin** role and is
audited server-side (`GET`/`PUT`/`DELETE /api/v1/agent-memory/governance`).

A policy runs in one of three modes: `MemoryGovernanceOff` (governance inert),
`MemoryGovernanceMonitor` (record would-be verdicts, never block a write), or
`MemoryGovernanceEnforce` (verdicts act — block / require approval).

```go
enabled := true
requireApproval := true
minConf := 0.3

// Upsert the org-wide policy (nil ProjectID = org scope; a *string scopes it
// to one project). Omitted fields keep their existing value / server default.
policy, err := client.SetAgentMemoryGovernance(ctx, evalguard.SetAgentMemoryGovernanceRequest{
    OrgID:   "22222222-2222-4222-8222-222222222222",
    Enabled: &enabled,
    Mode:    evalguard.MemoryGovernanceEnforce,
    Config: &evalguard.MemoryGovernanceConfig{
        Thresholds:               &evalguard.MemoryGovernanceThresholds{PoisonMinConfidence: &minConf},
        RequireApprovalOnRewrite: &requireApproval,
    },
})
if err != nil {
    log.Fatal(err)
}
log.Printf("policy %s mode=%s enabled=%v", policy.ID, policy.Mode, policy.Enabled)

// Read the project-scoped policy (returns nil, nil when none is configured).
projectID := "11111111-1111-4111-8111-111111111111"
got, err := client.GetAgentMemoryGovernance(ctx, policy.OrgID, &projectID)
if err != nil {
    log.Fatal(err)
}
if got == nil {
    log.Println("no project-scoped policy — org policy applies")
}

// Remove the policy (reverts to no governance); reports whether a row was removed.
removed, err := client.DeleteAgentMemoryGovernance(ctx, policy.OrgID, &projectID)
```

## Gateway Guardrail Config

Manage the per-project inline guardrails that run on the gateway hot path. Each
row enables **one** guardrail *vendor* on the project, ordered by `Priority`
(ascending). Upsert is idempotent on `(projectId, vendor)`. Upsert and delete
require **Admin** role, are audited, and bust the gateway loader cache so a
change takes effect on the very next proxy request
(`GET`/`POST`/`DELETE /api/v1/gateway/guardrails`).

A vendor is either a **partner adapter** (Lakera / Aporia / Patronus / … —
resolved server-side from a stored provider key referenced by `SecretRef`) or a
**local preset** that makes no external call and needs no secret. The four local
vendors are `local-firewall`, `moderated-firewall`, and the two Wave-2 agent
guardrails `data-not-instructions` and `tool-call-circuit-breaker`. The SDK
enforces the split client-side (matching the server's 400s): a local vendor must
**not** carry a `SecretRef`; every partner vendor **must**. Use
`evalguard.IsLocalGuardrailVendor(vendor)` / `evalguard.LocalGuardrailVendors` to
branch. On a flag, `OnFlag` selects the action: `GuardrailFlagBlock`,
`GuardrailFlagRedact`, or `GuardrailFlagFlag`.

```go
enabled := true

// Enable a LOCAL agent guardrail — no SecretRef (would 400 if supplied).
row, err := client.UpsertGuardrailConfig(ctx, evalguard.UpsertGuardrailConfigRequest{
    OrgID:     "22222222-2222-4222-8222-222222222222",
    ProjectID: "11111111-1111-4111-8111-111111111111",
    Vendor:    "tool-call-circuit-breaker",
    Config:    map[string]any{"maxRepeats": 3},
    OnFlag:    evalguard.GuardrailFlagFlag,
    Enabled:   &enabled,
})
if err != nil {
    log.Fatal(err)
}

// Enable a PARTNER vendor — SecretRef points at a stored provider key.
secretRef := "33333333-3333-4333-8333-333333333333"
_, err = client.UpsertGuardrailConfig(ctx, evalguard.UpsertGuardrailConfigRequest{
    OrgID:     row.OrgID,
    ProjectID: row.ProjectID,
    Vendor:    "lakera",
    SecretRef: &secretRef,
    OnFlag:    evalguard.GuardrailFlagBlock,
})

// List the project's rows (priority ascending; empty slice when none).
configs, err := client.ListGuardrailConfigs(ctx, row.ProjectID)
for _, g := range configs {
    log.Printf("%s on_flag=%s local=%v", g.Vendor, g.OnFlag, evalguard.IsLocalGuardrailVendor(g.Vendor))
}

// Delete one row by id (returns the deleted id).
deletedID, err := client.DeleteGuardrailConfig(ctx, row.ProjectID, row.ID)
```

> A multi-vendor `VendorChain` must lead with the primary `Vendor` (its
> `(projectId, vendor)` upsert key); the SDK rejects a chain whose first element
> differs before making the request.

## Importers

The Go SDK imports observability and corpus data into EvalGuard through the
ingest endpoints:

- **`CreateTrace(ctx, projectID, sessionID, steps)`** — create one trace from an
  ordered list of step maps (`POST /api/v1/traces`).
- **`IngestOTLP(ctx, resourceSpans)`** — import OpenTelemetry spans in OTLP
  `resourceSpans` shape (`POST /api/v1/ingest/otlp/traces`). Use this to forward
  spans from any OTel-instrumented app or collector.
- **`IngestShadowAISightings(ctx, source, rows, projectID)`** — import CASB /
  proxy / gateway logs for shadow-AI discovery (`POST /api/v1/shadow-ai/ingest`);
  `source` names the log origin.
- **`IngestRAGDocuments(ctx, req)`** — import a document corpus into the RAG
  pipeline, which chunks (and optionally embeds) each document and runs DLP +
  prompt-injection screening on every chunk (`POST /api/v1/rag/ingest`).
  Embedding uses the tenant's project-scoped BYOK OpenAI key.

```go
// OTLP: forward already-instrumented spans.
_, err := client.IngestOTLP(ctx, resourceSpans)

// RAG corpus import with recursive chunking + embedding.
_, err = client.IngestRAGDocuments(ctx, &evalguard.IngestRAGRequest{
    ProjectID: "11111111-1111-4111-8111-111111111111",
    Documents: []evalguard.RAGDocument{{Text: "…", Metadata: map[string]any{"src": "handbook"}}},
    Chunking:  &evalguard.RAGChunking{Strategy: "recursive", ChunkSize: 800, ChunkOverlap: 100},
    Embed:     true,
})
```

> Parsing **competitor trace-export formats** (Helicone, Langfuse, LangSmith,
> Braintrust, DeepEval, Ragas, Portkey, Humanloop, Vellum, Athina, Maxim,
> HuggingFace) is handled by `@evalguard/core`'s `importTraces` and the dashboard
> import UI, not by a Go method. Feed the resulting normalized spans to
> `IngestOTLP` / `CreateTrace` from Go.

Docs: https://evalguard.ai/docs/go-sdk

## License

Apache License, Version 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

This SDK is a thin public client for the EvalGuard service. The backend
engine, scorers, attack plugins, and proprietary logic are NOT covered by
Apache 2.0 — they are operated as a hosted service governed by the
[Terms of Service](https://evalguard.ai/terms).

"EvalGuard" is a trademark of EvalGuard, Inc.
