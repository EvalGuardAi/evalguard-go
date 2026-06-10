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

Docs: https://evalguard.ai/docs/go-sdk

## License

Apache License, Version 2.0 — see [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

This SDK is a thin public client for the EvalGuard service. The backend
engine, scorers, attack plugins, and proprietary logic are NOT covered by
Apache 2.0 — they are operated as a hosted service governed by the
[Terms of Service](https://evalguard.ai/terms).

"EvalGuard" is a trademark of EvalGuard, Inc.
