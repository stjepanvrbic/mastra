---
'@mastra/code-sdk': patch
'@mastra/factory': patch
---

Allow a host to supply the agent behind the session controller: `agent` on `MastraCodeConfig`, forwarded by `MastraFactoryConfig`. Mastra Code's controller, workspace, storage and session lifecycle are unchanged; only the agent the controller drives is swapped, so a host can run sessions on an SDK-backed agent (`@mastra/claude`, `@mastra/codex`) instead of the built-in code agent. Because the built-in agent resolves `extraTools` through its own dynamic tool set, an injected agent also receives them as controller toolsets, and the factory form `agent: ({ tools }) => Agent` hands them to an agent that resolves its own tools. Omitting `agent` keeps the built-in code agent, unchanged.
