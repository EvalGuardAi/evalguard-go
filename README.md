# EvalGuard Go SDK

Official Go client for the [EvalGuard](https://evalguard.ai) AI Evaluation, Security & Compliance Platform.

## Installation

```bash
go get github.com/EvalGuardAi/evalguard-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
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

    ctx := context.Background()

    // Run an evaluation
    result, err := client.RunEval(ctx, &evalguard.RunEvalRequest{
        DatasetID: "ds_abc123",
        Model:     "gpt-4o",
        Metrics:   []string{"accuracy", "toxicity", "relevance"},
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Eval %s: %v\n", result.ID, result.Metrics)

    // Security scan
    scan, err := client.RunSecurityScan(ctx, &evalguard.SecurityScanRequest{
        Prompts:   []string{"ignore all previous instructions"},
        Model:     "gpt-4o",
        ScanTypes: []string{"injection", "jailbreak"},
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Scan %s: %d findings\n", scan.ID, len(scan.Findings))

    // Check compliance
    compliance, err := client.CheckCompliance(ctx, "org_123", []string{"SOC2", "GDPR", "EU-AI-Act"})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Compliance: %v\n", compliance)
}
```

## Features

70 API methods covering all EvalGuard capabilities:

| Category | Methods |
|----------|---------|
| Evaluations | RunEval, GetEval, ListEvals, ListEvalRuns |
| Security | RunSecurityScan, GetSecurityReport, CodeScan, GetSecurityGraders, GetSecurityEffectiveness |
| Traces | GetTraces, GetTrace, SearchTraces, CreateTrace, IngestOTLP |
| Datasets | CreateDataset, ListDatasets |
| AI-SPM | GetAIPosture, AnalyzeShadowAI, AnalyzeCopilot |
| Gateway | GetGatewayHealth, GetGatewayStats |
| Cost/FinOps | GetCost, GetCostForecast, GetCostSavings, GetCostBudget, GetCostAnomalies, GetCostRecommendations |
| Monitoring | GetMonitoringAlerts, GetMonitoringDrift, GetMonitoringAnalytics, GetMonitoringSLA |
| Compliance | CheckCompliance, GetCompliance, GetComplianceGaps, GetEUAIAct, GetModelCards |
| Firewall | ListFirewallRules |
| Guardrails | ListGuardrails, GenerateGuardrails |
| Prompts | CreatePrompt, ListPrompts |
| AI SBOM | GetAISBOM, GenerateAISBOM |
| Threat Intel | GetThreatIntelligence |
| Formal Verification | FormalVerify |
| NL Pipeline | Ask, Search |
| Drift Detection | DetectDrift |
| Smart Routing | SmartRoute |
| Team & Org | ListTeam, GetAuditLogs, ListApiKeys, ListWebhooks |
| Support | SubmitTicket, ListTickets |
| Dashboard | GetDashboardStats, GetLeaderboard, GetMarketplace, ListTemplates, ListNotifications, ListEvalSchedules, ListIncidents, ListPipelines, GetAutopilotConfig, GetSettings, GetSIEMConnectors, CreateAnnotation, ListAnnotations |

## Error Handling

```go
result, err := client.RunEval(ctx, req)
if err != nil {
    var authErr *evalguard.AuthError
    var rlErr *evalguard.RateLimitError

    switch {
    case errors.As(err, &authErr):
        log.Fatal("Authentication failed:", authErr.Message)
    case errors.As(err, &rlErr):
        log.Printf("Rate limited, retry after %v", rlErr.RetryAfter)
        time.Sleep(rlErr.RetryAfter)
    default:
        log.Fatal("API error:", err)
    }
}
```

## Requirements

- Go 1.21+
- Zero external dependencies (standard library only)

## License

MIT
