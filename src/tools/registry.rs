//! Tool registry — stores tools and generates definitions for the LLM.
//!
//! The `ToolRegistry` is the central place where all available tools are
//! registered. It provides:
//! - Registration of tool instances (as `Box<dyn Tool>`)
//! - Lookup by name (for executing tool calls from the LLM)
//! - Generation of `ToolDef` arrays in the format Ollama/OpenAI expects
//!
//! # Why `HashMap<String, Box<dyn Tool>>`?
//!
//! We need heterogeneous storage — different tool types (FileReadTool,
//! ShellExecTool, etc.) in one collection. `Box<dyn Tool>` gives us that
//! via dynamic dispatch. The `HashMap` gives O(1) lookup by name, which
//! matters because the agent loop looks up tools on every tool call.

use std::collections::HashMap;

use tracing::{debug, info, warn};

use crate::agent::message::{FunctionDef, ToolDef};
use crate::tools::traits::Tool;

/// A collection of tools available to the agent.
///
/// Tools are registered once at startup and then used throughout the agent
/// loop. The registry is immutable after construction (no adding/removing
/// tools mid-conversation) — this keeps the tool definitions consistent
/// across LLM calls within a session.
pub struct ToolRegistry {
    /// Tools indexed by their unique name.
    tools: HashMap<String, Box<dyn Tool>>,

    /// Insertion order preserved for deterministic tool definition output.
    /// (HashMap iteration order is arbitrary; we want consistent ordering
    /// so the LLM sees the same tool list every time.)
    order: Vec<String>,
}

impl ToolRegistry {
    /// Create an empty tool registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Register a tool in the registry.
    ///
    /// The tool's `name()` must be unique. If a tool with the same name is
    /// already registered, the old one is replaced and a warning is logged.
    ///
    /// # Why `impl Tool + 'static`?
    ///
    /// The `impl Tool` accepts any concrete type that implements `Tool`.
    /// The `'static` bound is required because `Box<dyn Tool>` needs owned
    /// data (no borrowed references with limited lifetimes). This is standard
    /// for trait objects stored in collections.
    pub fn register(&mut self, tool: impl Tool + 'static) {
        let name = tool.name().to_string();

        if self.tools.contains_key(&name) {
            warn!(tool = %name, "Replacing existing tool with same name");
        } else {
            self.order.push(name.clone());
        }

        info!(
            tool = %name,
            description = %tool.description(),
            "Registered tool"
        );

        self.tools.insert(name, Box::new(tool));
    }

    /// Look up a tool by name.
    ///
    /// Returns `None` if no tool with that name is registered. The agent loop
    /// uses this to find the right tool when the LLM makes a tool call.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        let tool = self.tools.get(name).map(|b| b.as_ref());
        if tool.is_none() {
            debug!(tool = %name, "Tool not found in registry");
        }
        tool
    }

    /// Generate tool definitions in the format expected by Ollama/OpenAI.
    ///
    /// These `ToolDef` values are passed to `LlmBackend::chat()` so the LLM
    /// knows what tools are available and how to call them.
    ///
    /// The definitions are returned in registration order for determinism.
    pub fn tool_definitions(&self) -> Vec<ToolDef> {
        self.order
            .iter()
            .filter_map(|name| self.tools.get(name))
            .map(|tool| ToolDef {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.parameters_schema(),
                },
            })
            .collect()
    }

    /// Get the names of all registered tools (in registration order).
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// How many tools are registered.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::traits::ToolResult;
    use anyhow::Result;
    use serde_json::Value;
    use std::future::Future;
    use std::pin::Pin;

    /// Minimal test tool.
    struct FakeTool {
        tool_name: &'static str,
    }

    impl Tool for FakeTool {
        fn name(&self) -> &str {
            self.tool_name
        }

        fn description(&self) -> &str {
            "A fake tool for testing"
        }

        fn parameters_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string" }
                },
                "required": ["input"]
            })
        }

        fn execute(
            &self,
            _args: Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + '_>> {
            Box::pin(async move { Ok(ToolResult::success("fake output")) })
        }
    }

    #[test]
    fn test_register_and_lookup() {
        let mut registry = ToolRegistry::new();
        registry.register(FakeTool { tool_name: "alpha" });
        registry.register(FakeTool { tool_name: "beta" });

        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert!(registry.get("alpha").is_some());
        assert!(registry.get("beta").is_some());
        assert!(registry.get("gamma").is_none());
    }

    #[test]
    fn test_names_in_registration_order() {
        let mut registry = ToolRegistry::new();
        registry.register(FakeTool { tool_name: "zebra" });
        registry.register(FakeTool { tool_name: "apple" });
        registry.register(FakeTool { tool_name: "mango" });

        assert_eq!(registry.names(), &["zebra", "apple", "mango"]);
    }

    #[test]
    fn test_tool_definitions_format() {
        let mut registry = ToolRegistry::new();
        registry.register(FakeTool {
            tool_name: "test_tool",
        });

        let defs = registry.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].tool_type, "function");
        assert_eq!(defs[0].function.name, "test_tool");
        assert_eq!(defs[0].function.description, "A fake tool for testing");
        assert_eq!(defs[0].function.parameters["type"], "object");
    }

    #[test]
    fn test_replace_duplicate_name() {
        let mut registry = ToolRegistry::new();
        registry.register(FakeTool { tool_name: "dup" });
        registry.register(FakeTool { tool_name: "dup" });

        // Should only have one entry, not two
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.names().len(), 1);
    }

    #[test]
    fn test_empty_registry() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.tool_definitions().is_empty());
        assert!(registry.get("anything").is_none());
    }

    #[tokio::test]
    async fn test_execute_through_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(FakeTool {
            tool_name: "runner",
        });

        let tool = registry.get("runner").unwrap();
        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert_eq!(result.content, "fake output");
    }
}
