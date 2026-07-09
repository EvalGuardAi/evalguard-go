// Typed Prompt configuration + portable `.prompt` file format.
//
// Phase 1: a prompt version is a first-class TYPED artifact —
// a PromptConfig (model + decoding + tools), a structured template (completion
// string or ordered chat messages), and a portable `.prompt` file.
//
// Field names are the standard LLM generation parameters (model, temperature,
// topP, maxTokens, presencePenalty, seed, responseFormat, reasoningEffort,
// tools, etc.) shared across the major LLM APIs, and enum values are those
// APIs' industry-standard identifiers; both mirror EvalGuard's TypeScript core
// (packages/core/src/prompts/prompt-config.ts). The `.prompt` wire format is
// byte-compatible with the TS core serializer, so a `.prompt` file written by
// any EvalGuard SDK round-trips through every other.
//
// SECURITY: this serializes prompt CONFIG (metadata) only — no environment
// values, credentials or secrets are ever read or injected.

package evalguard

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Enum value slices — the industry-standard identifiers used by the major LLM
// APIs. Each is modeled as an open union (known members + Any), so these drive
// validation/autocomplete but an unknown member is never rejected (forward-compat).
var (
	ModelEndpointValues = []string{"complete", "chat", "edit"}

	ModelProviderValues = []string{
		"anthropic", "bedrock", "cohere", "deepseek", "google",
		"groq", "mock", "openai", "openai_azure", "replicate",
	}

	TemplateLanguageValues = []string{"default", "jinja"}

	OpenAIReasoningEffortValues = []string{"high", "medium", "low"}

	ResponseFormatTypeValues = []string{"json_object", "json_schema"}

	ChatRoleValues = []string{"user", "assistant", "system", "tool", "developer"}
)

// ResponseFormat is the response format of the model, e.g. {"type":"json_object"}.
type ResponseFormat struct {
	Type       string         `json:"type"`
	JSONSchema map[string]any `json:"jsonSchema,omitempty"`
}

// ToolFunction is a tool spec the model may choose to call.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Strict      *bool          `json:"strict,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// ChatTemplateMessage is a single role-tagged message in a chat template.
type ChatTemplateMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// PromptConfig is the typed superset of the previously-opaque prompt metadata.
// Its fields are the standard LLM generation parameters shared across the major
// LLM APIs. Only Model is required; every other field is an optional
// decoding/behaviour knob (pointer or nil-able so "unset" is distinct from a
// zero value). Stop and ReasoningEffort are unions (string|[]string and
// string|number respectively), modeled as `any`.
type PromptConfig struct {
	Model            string          `json:"model"`
	Endpoint         string          `json:"endpoint,omitempty"`
	Provider         string          `json:"provider,omitempty"`
	MaxTokens        *int            `json:"maxTokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"topP,omitempty"`
	Stop             any             `json:"stop,omitempty"`
	PresencePenalty  *float64        `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64        `json:"frequencyPenalty,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
	ResponseFormat   *ResponseFormat `json:"responseFormat,omitempty"`
	ReasoningEffort  any             `json:"reasoningEffort,omitempty"`
	Tools            []ToolFunction  `json:"tools,omitempty"`
	LinkedTools      []string        `json:"linkedTools,omitempty"`
	Attributes       map[string]any  `json:"attributes,omitempty"`
	Other            map[string]any  `json:"other,omitempty"`
}

// PromptTemplate is a completion string OR an ordered list of chat messages
// (exactly one is set). Build one with NewCompletionTemplate / NewChatTemplate.
type PromptTemplate struct {
	Text     *string
	Messages []ChatTemplateMessage
}

// NewCompletionTemplate builds a completion-string template.
func NewCompletionTemplate(text string) *PromptTemplate {
	return &PromptTemplate{Text: &text}
}

// NewChatTemplate builds a chat (message-list) template.
func NewChatTemplate(messages []ChatTemplateMessage) *PromptTemplate {
	return &PromptTemplate{Messages: messages}
}

// IsChatTemplate reports whether the template is a chat (message-list) template.
func IsChatTemplate(t *PromptTemplate) bool {
	return t != nil && t.Messages != nil
}

// PromptFile is the portable identity of a prompt version: config + template +
// template language.
type PromptFile struct {
	Config           PromptConfig
	Template         *PromptTemplate
	TemplateLanguage string
}

// configFieldOrder is the fixed frontmatter emit order — MUST match
// packages/core/src/prompts/prompt-file.ts CONFIG_FIELD_ORDER so serialized
// files are byte-identical across SDKs.
var configFieldOrder = []string{
	"model", "endpoint", "provider", "maxTokens", "temperature", "topP",
	"stop", "presencePenalty", "frequencyPenalty", "seed", "responseFormat",
	"reasoningEffort", "tools", "linkedTools", "attributes", "other",
}

var configKeySet = func() map[string]bool {
	m := make(map[string]bool, len(configFieldOrder))
	for _, k := range configFieldOrder {
		m[k] = true
	}
	return m
}()

var chatRoleSet = func() map[string]bool {
	m := make(map[string]bool, len(ChatRoleValues))
	for _, r := range ChatRoleValues {
		m[r] = true
	}
	return m
}()

const frontmatterDelimiter = "---"

// openTagRe matches a message open-tag line, e.g. `<system>` or `<user name="bob">`.
var openTagRe = regexp.MustCompile(`^<([a-zA-Z]+)(?:\s+name=("(?:[^"\\]|\\.)*"))?>$`)

// ValidatePromptConfig structurally validates a PromptConfig against the
// standard documented numeric ranges for these LLM parameters. Returns (valid,
// errors). Unknown enum members are permitted (open union). Mirrors the TS core
// validatePromptConfig.
func ValidatePromptConfig(c PromptConfig) (bool, []string) {
	var errs []string

	if strings.TrimSpace(c.Model) == "" {
		errs = append(errs, "`model` is required and must be a non-empty string")
	}
	if c.TopP != nil && (*c.TopP < 0 || *c.TopP > 1) {
		errs = append(errs, "`topP` must be a number between 0 and 1")
	}
	if c.PresencePenalty != nil && (*c.PresencePenalty < -2 || *c.PresencePenalty > 2) {
		errs = append(errs, "`presencePenalty` must be a number between -2.0 and 2.0")
	}
	if c.FrequencyPenalty != nil && (*c.FrequencyPenalty < -2 || *c.FrequencyPenalty > 2) {
		errs = append(errs, "`frequencyPenalty` must be a number between -2.0 and 2.0")
	}
	if c.MaxTokens != nil && *c.MaxTokens < -1 {
		errs = append(errs, "`maxTokens` must be an integer >= -1 (-1 = dynamic)")
	}
	if c.ReasoningEffort != nil && !isValidReasoningEffort(c.ReasoningEffort) {
		errs = append(errs, "`reasoningEffort` must be one of high|medium|low or a non-negative integer token budget")
	}
	for i, t := range c.Tools {
		if strings.TrimSpace(t.Name) == "" {
			errs = append(errs, fmt.Sprintf("tools[%d].name is required", i))
		}
	}
	if c.ResponseFormat != nil && c.ResponseFormat.Type == "" {
		errs = append(errs, "`responseFormat.type` is required")
	}

	return len(errs) == 0, errs
}

func isValidReasoningEffort(v any) bool {
	switch x := v.(type) {
	case string:
		for _, e := range OpenAIReasoningEffortValues {
			if x == e {
				return true
			}
		}
		return false
	case int:
		return x >= 0
	case int64:
		return x >= 0
	case float64:
		return x >= 0
	default:
		return false
	}
}

// SerializePromptFile serializes a PromptFile to the `.prompt` text format.
// Returns an error if the config is structurally invalid (e.g. missing model).
func SerializePromptFile(file PromptFile) (string, error) {
	if ok, errs := ValidatePromptConfig(file.Config); !ok {
		return "", fmt.Errorf("cannot serialize .prompt file — invalid config: %s", strings.Join(errs, "; "))
	}

	raw, err := marshalCompact(file.Config)
	if err != nil {
		return "", err
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return "", err
	}

	var lines []string
	for _, key := range configFieldOrder {
		if rv, ok := m[key]; ok {
			lines = append(lines, key+": "+string(rv))
		}
	}
	if file.TemplateLanguage != "" {
		tl, err := marshalCompact(file.TemplateLanguage)
		if err != nil {
			return "", err
		}
		lines = append(lines, "templateLanguage: "+string(tl))
	}
	frontmatter := strings.Join(lines, "\n")

	body, err := serializeBody(file.Template)
	if err != nil {
		return "", err
	}
	return frontmatterDelimiter + "\n" + frontmatter + "\n" + frontmatterDelimiter + "\n" + body, nil
}

func serializeBody(t *PromptTemplate) (string, error) {
	if t == nil {
		return "", nil
	}
	if IsChatTemplate(t) {
		parts := make([]string, len(t.Messages))
		for i, msg := range t.Messages {
			s, err := serializeMessage(msg)
			if err != nil {
				return "", err
			}
			parts[i] = s
		}
		return "\n" + strings.Join(parts, "\n") + "\n", nil
	}
	if t.Text != nil {
		return *t.Text, nil
	}
	return "", nil
}

func serializeMessage(m ChatTemplateMessage) (string, error) {
	nameAttr := ""
	if m.Name != "" {
		nb, err := marshalCompact(m.Name)
		if err != nil {
			return "", err
		}
		nameAttr = " name=" + string(nb)
	}
	return "<" + m.Role + nameAttr + ">\n" + m.Content + "\n</" + m.Role + ">", nil
}

// ParsePromptFile parses a `.prompt` file back into a PromptFile — the inverse
// of SerializePromptFile. Returns an error on malformed input.
func ParsePromptFile(text string) (*PromptFile, error) {
	opening := frontmatterDelimiter + "\n"
	if !strings.HasPrefix(text, opening) {
		return nil, fmt.Errorf(".prompt file must start with a `---` frontmatter delimiter")
	}
	term := "\n" + frontmatterDelimiter + "\n"
	termIdx := strings.Index(text[len(opening)-1:], term)
	if termIdx == -1 {
		return nil, fmt.Errorf(".prompt file is missing its closing `---` frontmatter delimiter")
	}
	termIdx += len(opening) - 1

	frontmatterBlock := text[len(opening):termIdx]
	body := text[termIdx+len(term):]

	config, templateLanguage, err := parseFrontmatter(frontmatterBlock)
	if err != nil {
		return nil, err
	}
	template, err := parseBody(body)
	if err != nil {
		return nil, err
	}
	return &PromptFile{Config: config, Template: template, TemplateLanguage: templateLanguage}, nil
}

func parseFrontmatter(block string) (PromptConfig, string, error) {
	m := map[string]json.RawMessage{}
	templateLanguage := ""

	if block != "" {
		for _, line := range strings.Split(block, "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			idx := strings.Index(line, ": ")
			if idx == -1 {
				return PromptConfig{}, "", fmt.Errorf("malformed .prompt frontmatter line: %q", line)
			}
			key := line[:idx]
			rawValue := line[idx+2:]
			if !json.Valid([]byte(rawValue)) {
				return PromptConfig{}, "", fmt.Errorf("malformed .prompt frontmatter value for %q: %s", key, rawValue)
			}
			if key == "templateLanguage" {
				if err := json.Unmarshal([]byte(rawValue), &templateLanguage); err != nil {
					return PromptConfig{}, "", err
				}
			} else if configKeySet[key] {
				m[key] = json.RawMessage(rawValue)
			} else {
				return PromptConfig{}, "", fmt.Errorf("unknown .prompt frontmatter field: %q", key)
			}
		}
	}

	if _, ok := m["model"]; !ok {
		return PromptConfig{}, "", fmt.Errorf(".prompt frontmatter is missing required `model` field")
	}

	reassembled, err := json.Marshal(m)
	if err != nil {
		return PromptConfig{}, "", err
	}
	var config PromptConfig
	if err := json.Unmarshal(reassembled, &config); err != nil {
		return PromptConfig{}, "", err
	}
	return config, templateLanguage, nil
}

func parseBody(body string) (*PromptTemplate, error) {
	if body == "" {
		return nil, nil
	}
	for _, line := range strings.Split(body, "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		if role, _, ok := parseOpenTag(line); ok && chatRoleSet[role] {
			msgs, err := parseChatBody(body)
			if err != nil {
				return nil, err
			}
			return &PromptTemplate{Messages: msgs}, nil
		}
		break
	}
	return &PromptTemplate{Text: &body}, nil
}

func parseOpenTag(line string) (role string, name string, ok bool) {
	match := openTagRe.FindStringSubmatch(line)
	if match == nil {
		return "", "", false
	}
	role = match[1]
	if match[2] != "" {
		if err := json.Unmarshal([]byte(match[2]), &name); err != nil {
			return "", "", false
		}
	}
	return role, name, true
}

func parseChatBody(body string) ([]ChatTemplateMessage, error) {
	lines := strings.Split(body, "\n")
	var messages []ChatTemplateMessage

	i := 0
	for i < len(lines) {
		line := lines[i]
		if strings.TrimSpace(line) == "" {
			i++
			continue
		}
		role, name, ok := parseOpenTag(line)
		if !ok || !chatRoleSet[role] {
			return nil, fmt.Errorf("unexpected line in .prompt chat body: %q", line)
		}
		closeTag := "</" + role + ">"
		var contentLines []string
		i++
		closed := false
		for i < len(lines) {
			if lines[i] == closeTag {
				closed = true
				i++
				break
			}
			contentLines = append(contentLines, lines[i])
			i++
		}
		if !closed {
			return nil, fmt.Errorf("unterminated <%s> block in .prompt chat body", role)
		}
		messages = append(messages, ChatTemplateMessage{
			Role:    role,
			Content: strings.Join(contentLines, "\n"),
			Name:    name,
		})
	}
	return messages, nil
}

// PromptFileFrom builds a PromptFile from typed artifact fields; returns an
// error if no typed config is present (a content-only version has no portable
// `.prompt` representation).
func PromptFileFrom(config *PromptConfig, template *PromptTemplate, templateLanguage string) (*PromptFile, error) {
	if config == nil {
		return nil, fmt.Errorf("cannot build a .prompt file from a version without a typed `config` (model/decoding settings)")
	}
	return &PromptFile{Config: *config, Template: template, TemplateLanguage: templateLanguage}, nil
}

// marshalCompact JSON-encodes v with compact separators and HTML escaping
// disabled, so output matches JS JSON.stringify byte-for-byte.
func marshalCompact(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	b := buf.Bytes()
	if len(b) > 0 && b[len(b)-1] == '\n' {
		b = b[:len(b)-1]
	}
	return b, nil
}
