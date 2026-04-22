# EvalGuard Go SDK

[![Go Reference](https://pkg.go.dev/badge/github.com/EvalGuardAi/evalguard-go.svg)](https://pkg.go.dev/github.com/EvalGuardAi/evalguard-go)

Go client for [EvalGuard](https://evalguard.ai) — LLM evaluation, red-team
testing, and runtime guardrails.

## Install

```bash
go get github.com/EvalGuardAi/evalguard-go@latest
```

## Quick start

```go
package main

import (
    "context"
    "log"
    "time"

    "github.com/EvalGuardAi/evalguard-go"
)

func main() {
    client, err := evalguard.NewClient("your-api-key",
        evalguard.WithBaseURL("https://api.evalguard.ai"),
        evalguard.WithTimeout(30*time.Second),
    )
    if err != nil {
        log.Fatal(err)
    }

    result, err := client.RunEval(context.Background(), &evalguard.RunEvalRequest{
        DatasetID: "ds_abc123",
        Model:     "gpt-4o",
        Metrics:   []string{"accuracy", "toxicity", "relevance"},
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("%+v", result)
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
