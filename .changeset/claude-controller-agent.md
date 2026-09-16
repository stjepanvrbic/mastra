---
'@mastra/claude': minor
---

Add `ClaudeCodeControllerAgent`, a `ClaudeSDKAgent` that satisfies `AgentController`, so a controller-driven product (Mastra Code, Factory) can run its sessions on the Claude Code binary instead of the built-in code agent. It registers its run with the thread runtime the way `Agent.stream` does (so the run is pumped, observed as finished, and the thread's reservation clears), bridges the controller's toolsets into one in-process MCP server (`ask_user`, `submit_plan` and the host's own tools become callable from the session), turns a tool's `suspend()` into `tool-call-suspended` and resumes it from `sendStreamResume`, mirrors native tool activity as chunks and writes each turn back to the thread, and honours `untilIdle` by making signals queued during a turn the next input of the same run and the same Claude session. One Claude session per thread. `ClaudeSDKAgent` is unchanged.
