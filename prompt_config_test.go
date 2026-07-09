package evalguard

import (
	"reflect"
	"testing"
)

func ptrInt(i int) *int           { return &i }
func ptrFloat(f float64) *float64 { return &f }

func TestEnumValuesMatchStandard(t *testing.T) {
	if !reflect.DeepEqual(ModelEndpointValues, []string{"complete", "chat", "edit"}) {
		t.Fatalf("ModelEndpointValues mismatch: %v", ModelEndpointValues)
	}
	if !reflect.DeepEqual(ModelProviderValues, []string{
		"anthropic", "bedrock", "cohere", "deepseek", "google",
		"groq", "mock", "openai", "openai_azure", "replicate",
	}) {
		t.Fatalf("ModelProviderValues mismatch: %v", ModelProviderValues)
	}
	if !reflect.DeepEqual(TemplateLanguageValues, []string{"default", "jinja"}) {
		t.Fatalf("TemplateLanguageValues mismatch")
	}
	if !reflect.DeepEqual(OpenAIReasoningEffortValues, []string{"high", "medium", "low"}) {
		t.Fatalf("OpenAIReasoningEffortValues mismatch")
	}
	if !reflect.DeepEqual(ResponseFormatTypeValues, []string{"json_object", "json_schema"}) {
		t.Fatalf("ResponseFormatTypeValues mismatch")
	}
	if !reflect.DeepEqual(ChatRoleValues, []string{"user", "assistant", "system", "tool", "developer"}) {
		t.Fatalf("ChatRoleValues mismatch")
	}
}

func TestValidatePromptConfig(t *testing.T) {
	if ok, _ := ValidatePromptConfig(PromptConfig{Model: "gpt-4o", Temperature: ptrFloat(0.7)}); !ok {
		t.Fatal("expected valid config")
	}
	ok, errs := ValidatePromptConfig(PromptConfig{
		Model:           "",
		TopP:            ptrFloat(2),
		PresencePenalty: ptrFloat(5),
		MaxTokens:       ptrInt(-3),
	})
	if ok {
		t.Fatal("expected invalid config")
	}
	if len(errs) < 4 {
		t.Fatalf("expected >=4 errors, got %d: %v", len(errs), errs)
	}
}

func TestReasoningEffortValidation(t *testing.T) {
	if ok, _ := ValidatePromptConfig(PromptConfig{Model: "o3", ReasoningEffort: "high"}); !ok {
		t.Fatal("enum reasoning effort should be valid")
	}
	if ok, _ := ValidatePromptConfig(PromptConfig{Model: "c", ReasoningEffort: 2048}); !ok {
		t.Fatal("int reasoning effort should be valid")
	}
	if ok, _ := ValidatePromptConfig(PromptConfig{Model: "c", ReasoningEffort: "turbo"}); ok {
		t.Fatal("unknown reasoning effort should be invalid")
	}
}

func TestIsChatTemplate(t *testing.T) {
	if !IsChatTemplate(NewChatTemplate([]ChatTemplateMessage{{Role: "user", Content: "hi"}})) {
		t.Fatal("expected chat template")
	}
	if IsChatTemplate(NewCompletionTemplate("just text")) {
		t.Fatal("completion template is not a chat template")
	}
}

func TestCanonicalCrossLanguageBytes(t *testing.T) {
	// MUST equal the byte layout produced by the TS core / Python serializers.
	file := PromptFile{
		Config: PromptConfig{
			Model:       "gpt-4o",
			Provider:    "openai",
			MaxTokens:   ptrInt(1024),
			Temperature: ptrFloat(0.7),
		},
		Template: NewChatTemplate([]ChatTemplateMessage{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "{{question}}"},
		}),
	}
	expected := "---\n" +
		"model: \"gpt-4o\"\n" +
		"provider: \"openai\"\n" +
		"maxTokens: 1024\n" +
		"temperature: 0.7\n" +
		"---\n" +
		"\n" +
		"<system>\nYou are a helpful assistant.\n</system>\n" +
		"<user>\n{{question}}\n</user>\n"
	got, err := SerializePromptFile(file)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	if got != expected {
		t.Fatalf("byte layout mismatch:\n got=%q\nwant=%q", got, expected)
	}
}

func TestRoundTripChat(t *testing.T) {
	strict := true
	file := PromptFile{
		Config: PromptConfig{
			Model:          "gpt-4o",
			Provider:       "openai",
			MaxTokens:      ptrInt(1024),
			Temperature:    ptrFloat(0.7),
			ResponseFormat: &ResponseFormat{Type: "json_object"},
			Tools: []ToolFunction{
				{Name: "search", Description: "Search the web", Strict: &strict},
			},
			LinkedTools: []string{"tool_123"},
		},
		Template: NewChatTemplate([]ChatTemplateMessage{
			{Role: "system", Content: "You are a helpful assistant.", Name: "alice"},
			{Role: "user", Content: "{{question}}"},
		}),
		TemplateLanguage: "jinja",
	}
	text, err := SerializePromptFile(file)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	parsed, err := ParsePromptFile(text)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// Serialize-parse-serialize idempotency is the strongest equality check that
	// is robust to union re-typing (Stop/ReasoningEffort as `any`).
	reText, err := SerializePromptFile(*parsed)
	if err != nil {
		t.Fatalf("re-serialize: %v", err)
	}
	if reText != text {
		t.Fatalf("round-trip not idempotent:\n first=%q\nsecond=%q", text, reText)
	}
	// Spot-check rehydrated fields.
	if parsed.Config.Model != "gpt-4o" || parsed.TemplateLanguage != "jinja" {
		t.Fatalf("bad rehydration: %+v", parsed.Config)
	}
	if len(parsed.Template.Messages) != 2 || parsed.Template.Messages[0].Name != "alice" {
		t.Fatalf("bad template rehydration: %+v", parsed.Template)
	}
	if parsed.Config.ResponseFormat == nil || parsed.Config.ResponseFormat.Type != "json_object" {
		t.Fatalf("bad response format: %+v", parsed.Config.ResponseFormat)
	}
}

func TestRoundTripCompletion(t *testing.T) {
	file := PromptFile{
		Config:   PromptConfig{Model: "gpt-3.5-turbo-instruct", Endpoint: "complete"},
		Template: NewCompletionTemplate("Summarize: {{doc}}"),
	}
	text, err := SerializePromptFile(file)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	parsed, err := ParsePromptFile(text)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if parsed.Template == nil || parsed.Template.Text == nil || *parsed.Template.Text != "Summarize: {{doc}}" {
		t.Fatalf("bad completion round-trip: %+v", parsed.Template)
	}
	if parsed.Config.Endpoint != "complete" {
		t.Fatalf("endpoint lost: %q", parsed.Config.Endpoint)
	}
}

func TestSerializeRejectsInvalidConfig(t *testing.T) {
	if _, err := SerializePromptFile(PromptFile{Config: PromptConfig{Model: ""}}); err == nil {
		t.Fatal("expected error serializing invalid config")
	}
}

func TestParseRejectsMalformed(t *testing.T) {
	if _, err := ParsePromptFile("no frontmatter here"); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestPromptFileFrom(t *testing.T) {
	cfg := PromptConfig{Model: "claude-3-5-sonnet"}
	pf, err := PromptFileFrom(&cfg, nil, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pf.Config.Model != "claude-3-5-sonnet" {
		t.Fatalf("bad config: %+v", pf.Config)
	}
	if _, err := PromptFileFrom(nil, nil, ""); err == nil {
		t.Fatal("expected error for nil config")
	}
}
