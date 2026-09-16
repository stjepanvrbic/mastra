---
'@mastra/core': patch
---

Export `agentThreadStreamRuntime` from `@mastra/core/agent`. An agent that overrides `stream()` to drive an external harness (the SDK-backed agents in `agent-sdks/*`) is bound to a thread by `AgentController` exactly like a built-in agent, but could not register its run with the thread runtime from outside the package: the run was never pumped, never observed as finished, and the thread's reservation never cleared, so every later message on that thread queued indefinitely. The runtime is now reachable, so such an agent can call `waitForCrossAgentThreadRun`, `prepareRunOptions`, `registerRun`, `drainPendingSignals` and `releaseThreadRunReservation` around its own loop.
