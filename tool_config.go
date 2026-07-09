// Typed Tool configuration + named-environment values (Phase 2).
//
// A sibling of prompt_config.go. A tool version is a typed artifact — a callable
// function spec plus source code, setup values and attributes — with an
// environment model (EnvironmentTag = "default" | "other"). The tool's
// callable Function REUSES the ToolFunction type already defined in
// prompt_config.go — it is NOT redefined. Mirrors the TypeScript core
// (packages/core/src/tools/tool-config.ts and prompts/environments.ts).

package evalguard

import "strings"

// EnvironmentTagValues is the environment-tag open union
// (Union[Literal["default","other"], Any]). "default" marks the fallback env.
var EnvironmentTagValues = []string{"default", "other"}

// SeededEnvironment is a named environment seeded for back-compat.
type SeededEnvironment struct {
	Name string
	Tag  string
}

// SeededEnvironments are the two environments that used to be the hardcoded
// "staging" | "production" union. "production" is seeded as the "default"
// (fallback) environment.
var SeededEnvironments = []SeededEnvironment{
	{Name: "production", Tag: "default"},
	{Name: "staging", Tag: "other"},
}

// ToolEnvironmentVariable is a managed Tool's runtime env var
// (a name/value pair).
type ToolEnvironmentVariable struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// ToolConfig is the version-identity of a managed Tool (function / source code /
// setup schema / attributes). Every field is optional; a version must
// carry at least a Function spec or SourceCode so it has a callable identity.
// Function REUSES the ToolFunction type from prompt_config.go.
type ToolConfig struct {
	Function    *ToolFunction  `json:"function,omitempty"`
	SourceCode  string         `json:"sourceCode,omitempty"`
	SetupSchema map[string]any `json:"setupSchema,omitempty"`
	Attributes  map[string]any `json:"attributes,omitempty"`
}

// ValidateToolConfig structurally validates a ToolConfig. Returns (valid,
// errors). Mirrors the TS core validateToolConfig: a version requires at least
// a Function spec or SourceCode.
func ValidateToolConfig(c ToolConfig) (bool, []string) {
	var errs []string

	if c.Function == nil && c.SourceCode == "" {
		errs = append(errs, "A Tool version requires at least a `function` spec or `sourceCode`")
	}
	if c.Function != nil {
		if strings.TrimSpace(c.Function.Name) == "" {
			errs = append(errs, "`function.name` is required and must be a non-empty string")
		}
	}

	return len(errs) == 0, errs
}
