import type { Query, SDKMessage } from '@anthropic-ai/claude-agent-sdk';
import { createTool } from '@mastra/core/tools';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import { ClaudeCodeControllerAgent } from './index';

const queryMock = vi.hoisted(() => vi.fn());
const createSdkMcpServerMock = vi.hoisted(() => vi.fn((config: unknown) => ({ type: 'sdk', ...(config as object) })));

vi.mock('@anthropic-ai/claude-agent-sdk', () => ({
  query: queryMock,
  createSdkMcpServer: createSdkMcpServerMock,
  tool: (name: string, description: string, inputSchema: unknown, handler: unknown) => ({ name, description, inputSchema, handler }),
}));

const MODEL = 'claude-sonnet-4-6';

function initMessage(sessionId: string): SDKMessage {
  return { type: 'system', subtype: 'init', session_id: sessionId, model: MODEL, cwd: '/tmp/project' } as SDKMessage;
}

function resultMessage(result: string): SDKMessage {
  return {
    type: 'result',
    subtype: 'success',
    duration_ms: 1,
    duration_api_ms: 1,
    is_error: false,
    num_turns: 1,
    result,
    stop_reason: 'end_turn',
    total_cost_usd: 0,
    permission_denials: [],
    uuid: 'message-uuid',
    session_id: 'session-id',
  } as SDKMessage;
}

/** A session that reports its init, then holds until `release()` before reporting its result. */
function gatedQuery(sessionId: string, gate: Promise<void>): Query {
  return (async function* () {
    yield initMessage(sessionId);
    await gate;
    yield resultMessage('done');
  })() as Query;
}

function plainQuery(sessionId: string): Query {
  return (async function* () {
    yield initMessage(sessionId);
    yield resultMessage('done');
  })() as Query;
}

function gate(): { gate: Promise<void>; release: () => void } {
  let release!: () => void;
  const promise = new Promise<void>(resolve => (release = resolve));
  return { gate: promise, release };
}

/** The SDK options `query()` was called with on the nth call. */
function sdkOptions(call = 0): Record<string, any> {
  return queryMock.mock.calls[call]?.[0]?.options ?? {};
}

/** The bridged tools handed to the in-process MCP server on the nth call. */
function mcpTools(call = 0): Array<{ name: string; handler: (args: unknown, extra: unknown) => Promise<any> }> {
  return (sdkOptions(call).mcpServers?.mastra as { tools?: any[] })?.tools ?? [];
}

function agentWith(options: Record<string, unknown> = {}) {
  return new ClaudeCodeControllerAgent({
    id: 'stage',
    description: 'A controller-driven Claude Code session',
    sdkOptions: { model: MODEL },
    ...options,
  } as ConstructorParameters<typeof ClaudeCodeControllerAgent>[0]);
}

/** Consumes the run's chunks in the background, the way the thread runtime's pump does. */
function consume(output: { fullStream: unknown }): { chunks: any[]; done: Promise<void> } {
  const chunks: any[] = [];
  const done = (async () => {
    for await (const chunk of output.fullStream as AsyncIterable<any>) chunks.push(chunk);
  })();
  return { chunks, done };
}

const untilQueried = (call = 0) => vi.waitFor(() => expect(queryMock.mock.calls.length).toBeGreaterThan(call));

describe('ClaudeCodeControllerAgent', () => {
  beforeEach(() => {
    queryMock.mockReset();
    createSdkMcpServerMock.mockClear();
  });

  it('bridges the controller built-ins and the agent-owned tools into one in-process MCP server', async () => {
    queryMock.mockImplementationOnce(() => plainQuery('s1'));
    const transition = createTool({
      id: 'transition',
      description: 'Move the card',
      inputSchema: z.object({ stage: z.string() }),
      execute: async () => 'moved',
    });
    const output = await agentWith({ tools: { transition } }).stream('go');
    const { done } = consume(output);
    await done;

    expect(mcpTools().map(t => t.name)).toEqual(expect.arrayContaining(['ask_user', 'submit_plan', 'transition']));
    expect(sdkOptions().allowedTools).toEqual(expect.arrayContaining(['mcp__mastra__transition', 'mcp__mastra__ask_user']));
    expect(createSdkMcpServerMock).toHaveBeenCalledOnce();
  });

  it('resolves tools from a function and skips the ones a Claude Code session already has', async () => {
    queryMock.mockImplementationOnce(() => plainQuery('s1'));
    const own = createTool({ id: 'own', description: 'own', inputSchema: z.object({}), execute: async () => 'ok' });
    const subagent = createTool({ id: 'subagent', description: 'no', inputSchema: z.object({}), execute: async () => 'no' });
    const output = await agentWith({ tools: () => ({ own, subagent }) }).stream('go');
    const { done } = consume(output);
    await done;

    const names = mcpTools().map(t => t.name);
    expect(names).toContain('own');
    expect(names).not.toContain('subagent');
  });

  it('runs a bridged tool during the session and mirrors its call and result as chunks', async () => {
    const { gate: held, release } = gate();
    queryMock.mockImplementationOnce(() => gatedQuery('s1', held));
    const execute = vi.fn(async () => ({ ok: true }));
    const transition = createTool({ id: 'transition', description: 'Move', inputSchema: z.object({ stage: z.string() }), execute });
    const output = await agentWith({ tools: { transition } }).stream('go');
    const { chunks, done } = consume(output);
    await untilQueried();

    const bridged = mcpTools().find(t => t.name === 'transition')!;
    await expect(bridged.handler({ stage: 'plan' }, undefined)).resolves.toEqual({
      content: [{ type: 'text', text: JSON.stringify({ ok: true }) }],
    });
    release();
    await done;

    expect(execute).toHaveBeenCalledWith(
      { stage: 'plan' },
      expect.objectContaining({ agent: expect.objectContaining({ agentId: 'stage' }) }),
    );
    expect(chunks.find(c => c.type === 'tool-call')?.payload).toMatchObject({ toolName: 'transition', args: { stage: 'plan' } });
    expect(chunks.find(c => c.type === 'tool-result')?.payload).toMatchObject({ toolName: 'transition', result: { ok: true } });
  });

  it('a suspending tool parks until sendStreamResume answers it, then re-runs with the resume data', async () => {
    const { gate: held, release } = gate();
    queryMock.mockImplementationOnce(() => gatedQuery('s1', held));
    const seen: unknown[] = [];
    const approval = createTool({
      id: 'approval',
      description: 'Ask first',
      inputSchema: z.object({}),
      execute: async (_args: unknown, context: any) => {
        seen.push(context.agent.resumeData);
        if (context.agent.resumeData === undefined) await context.agent.suspend({ question: 'ok?' });
        return `answered: ${(context.agent.resumeData as { answer: string }).answer}`;
      },
    });
    const agent = agentWith({ tools: { approval } });
    const output = await agent.stream('go');
    const { chunks, done } = consume(output);
    await untilQueried();

    const call = mcpTools().find(t => t.name === 'approval')!.handler({}, undefined);
    const suspended = await vi.waitFor(() => {
      const chunk = chunks.find(c => c.type === 'tool-call-suspended');
      expect(chunk).toBeDefined();
      return chunk!;
    });
    expect(suspended.payload).toMatchObject({ toolName: 'approval', suspendPayload: { question: 'ok?' } });

    await agent.sendStreamResume({ toolCallId: suspended.payload.toolCallId, resumeData: { answer: 'yes' } } as never);
    await expect(call).resolves.toEqual({ content: [{ type: 'text', text: 'answered: yes' }] });
    expect(seen).toEqual([undefined, { answer: 'yes' }]);
    release();
    await done;
  });

  it('refuses a resume for a tool call it does not hold', async () => {
    await expect(agentWith().sendStreamResume({ toolCallId: 'nope', resumeData: {} } as never)).rejects.toThrow(
      'no suspended tool call',
    );
  });

  it('keeps one Claude session per thread: a new session id first, resume after', async () => {
    queryMock.mockImplementation(() => plainQuery('session-1'));
    const agent = agentWith();
    for (const [prompt, thread] of [
      ['one', 'thread-a'],
      ['two', 'thread-a'],
      ['three', 'thread-b'],
    ]) {
      const output = await agent.stream(prompt, { memory: { thread, resource: 'res' } } as never);
      await consume(output).done;
    }

    expect(sdkOptions(0).sessionId).toEqual(expect.any(String));
    expect(sdkOptions(0).resume).toBeUndefined();
    expect(sdkOptions(1).resume).toBe('session-1');
    expect(sdkOptions(2).resume).toBeUndefined();
  });

  it('resolves the working directory through resolveCwd before the session starts', async () => {
    queryMock.mockImplementationOnce(() => plainQuery('s1'));
    const resolveCwd = vi.fn(async () => process.cwd());
    const output = await agentWith({ resolveCwd }).stream('go', { memory: { thread: 'thread-a' } } as never);
    await consume(output).done;

    expect(resolveCwd).toHaveBeenCalledWith('thread-a');
    expect(sdkOptions().cwd).toBe(process.cwd());
  });

  it('fails the run when the session reports a failure', async () => {
    queryMock.mockImplementationOnce(
      () =>
        (async function* () {
          yield initMessage('s1');
          yield { type: 'result', subtype: 'error_during_execution', errors: ['rate limited'] } as SDKMessage;
        })() as Query,
    );
    const output = await agentWith().stream('go');
    const { chunks, done } = consume(output);
    await done;

    expect(chunks.find(c => c.type === 'error')?.payload.error).toEqual(expect.objectContaining({ message: 'rate limited' }));
  });
});
