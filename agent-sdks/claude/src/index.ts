import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { ReadableStream } from 'node:stream/web';
import { setTimeout as sleep } from 'node:timers/promises';

import { createSdkMcpServer, query, tool as sdkTool } from '@anthropic-ai/claude-agent-sdk';
import type { ModelUsage, Options as ClaudeQueryOptions, SDKMessage } from '@anthropic-ai/claude-agent-sdk';

import { Agent, agentThreadStreamRuntime } from '@mastra/core/agent';
import type { SendAgentStreamResumeOptions, SendAgentStreamResumeResult, ToolsInput } from '@mastra/core/agent';
import type { MastraDBMessage, MessageListInput } from '@mastra/core/agent/message-list';
import { askUserTool, submitPlanTool } from '@mastra/core/agent-controller';
import type { Mastra } from '@mastra/core/mastra';
import type { CostContext } from '@mastra/core/observability';
import { RequestContext } from '@mastra/core/request-context';
import type {
  ChunkType,
  FullOutput,
  LanguageModelUsage,
  ProviderMetadata,
  MastraModelOutput,
} from '@mastra/core/stream';
import { ChunkFrom } from '@mastra/core/stream';
import {
  createMastraOutput,
  createNoopModel,
  createProviderMetadata,
  createSDKAgentTelemetry,
  enqueueFinishChunks,
  enqueueStartChunks,
  enqueueTextDelta,
  getStructuredOutputFromValue,
  getStructuredOutputSchema,
  promptToText,
  sumDefined,
  toFullOutput,
  toLanguageModelUsage,
} from './utils';
import type { SDKAgentRunOptions, SDKAgentTelemetry, SDKModelGenerateResult, V3Usage } from './utils';

const PROVIDER = '@anthropic-ai/claude-agent-sdk';
const MODEL_ID = 'claude-agent-sdk';

type ClaudeUsageTotals = {
  inputTokens?: number;
  outputTokens?: number;
  cacheReadInputTokens?: number;
  cacheCreationInputTokens?: number;
  totalCostUsd?: number;
  modelUsage?: Record<string, ModelUsage>;
};

export type ClaudeSDKOptions = ClaudeQueryOptions;

export type ClaudeAgentOptions = {
  /**
   * Mastra agent id used when registering this wrapper with Mastra.
   */
  id: string;
  /**
   * Optional display name for the Mastra agent. Defaults to `id`.
   */
  name?: string;
  /**
   * Description surfaced by Mastra when listing or selecting agents.
   */
  description: string;
  /**
   * Claude Agent SDK options forwarded to `query()` on every run.
   */
  sdkOptions?: ClaudeSDKOptions;
};

export type ClaudeSDKAgentRunOptions<OUTPUT = unknown> = SDKAgentRunOptions<OUTPUT> & {
  /**
   * Claude Agent SDK query options for this run. These are merged over the
   * wrapper's `sdkOptions`, which allows per-call session continuation options
   * such as `resume` or `continue`.
   */
  sdkOptions?: ClaudeSDKOptions;
};

export type ClaudeSDKAgentResumeData =
  | {
      /**
       * Message to send while resuming the Claude SDK session.
       */
      message: MessageListInput;
      /**
       * Claude session id to resume.
       */
      sessionId: string;
      /**
       * Fork the resumed session into a new session.
       */
      forkSession?: boolean;
      /**
       * Resume the session up to a specific assistant message UUID.
       */
      resumeSessionAt?: string;
    }
  | {
      /**
       * Message to send while continuing the latest Claude SDK session.
       */
      message: MessageListInput;
      /**
       * Continue the latest Claude session in the current working directory.
       */
      continue: true;
    };

export class ClaudeSDKAgent extends Agent {
  readonly options: ClaudeAgentOptions;
  #mastra?: Mastra;

  constructor(options: ClaudeAgentOptions) {
    super({
      id: options.id,
      name: options.name ?? options.id,
      description: options.description,
      instructions: '',
      model: createNoopModel({
        modelId: getModelId(options),
        provider: PROVIDER,
      }),
    });
    this.options = options;
  }

  override __registerMastra(mastra: Mastra): void {
    super.__registerMastra(mastra);
    this.#mastra = mastra;
  }

  supportsMemory(): boolean {
    return false;
  }

  async generate<OUTPUT = undefined>(
    messages: MessageListInput,
    options?: ClaudeSDKAgentRunOptions<OUTPUT>,
  ): Promise<FullOutput<OUTPUT>> {
    const prompt = promptToText(messages);
    const runId = options?.runId ?? randomUUID();
    const requestContext = options?.requestContext ?? new RequestContext();
    const instructions = options?.instructions ? promptToText(options.instructions) : undefined;
    const telemetry = createSDKAgentTelemetry({
      agentId: this.id,
      agentName: this.name,
      provider: PROVIDER,
      modelId: getModelId(this.options),
      messages,
      prompt,
      runId,
      streaming: false,
      method: 'generate',
      requestContext,
      instructions,
      maxSteps: options?.maxSteps,
      tracingOptions: options?.tracingOptions,
      tracingContext: options?.tracingContext,
      onFinish: options?.onFinish,
      onStepFinish: options?.onStepFinish,
      mastra: this.#mastra,
    });
    let result: SDKModelGenerateResult;
    try {
      result = await telemetry.execute(() => runClaudeGenerate(prompt, this.options, telemetry, options));
      telemetry.endGenerate(result);
    } catch (error) {
      telemetry.fail(error);
      throw error;
    }

    return toFullOutput<OUTPUT>({
      messages,
      runId,
      provider: PROVIDER,
      result,
      options: { ...telemetry.outputOptions(), structuredOutput: options?.structuredOutput as any },
    });
  }

  async stream<OUTPUT = undefined>(
    messages: MessageListInput,
    options?: ClaudeSDKAgentRunOptions<OUTPUT>,
  ): Promise<MastraModelOutput<OUTPUT>> {
    const runId = options?.runId ?? randomUUID();
    const prompt = promptToText(messages);
    const modelId = getModelId(this.options);
    const requestContext = options?.requestContext ?? new RequestContext();
    const instructions = options?.instructions ? promptToText(options.instructions) : undefined;
    const telemetry = createSDKAgentTelemetry({
      agentId: this.id,
      agentName: this.name,
      provider: PROVIDER,
      modelId,
      messages,
      prompt,
      runId,
      streaming: true,
      method: 'stream',
      requestContext,
      instructions,
      maxSteps: options?.maxSteps,
      tracingOptions: options?.tracingOptions,
      tracingContext: options?.tracingContext,
      onFinish: options?.onFinish,
      onStepFinish: options?.onStepFinish,
      mastra: this.#mastra,
    });

    return createMastraOutput<OUTPUT>({
      messages,
      runId,
      modelId,
      provider: PROVIDER,
      stream: telemetry.wrapStream(runClaudeAsMastraStream(prompt, this.options, runId, telemetry, options)),
      options: { ...telemetry.outputOptions(), structuredOutput: options?.structuredOutput as any },
    });
  }

  async resumeGenerate<OUTPUT = undefined>(
    resumeData: ClaudeSDKAgentResumeData,
    options?: ClaudeSDKAgentRunOptions<OUTPUT>,
  ): Promise<FullOutput<OUTPUT>> {
    const data = validateClaudeResumeData(resumeData);
    return this.generate(data.message, createClaudeResumeRunOptions(data, options));
  }

  async resumeStream<OUTPUT = undefined>(
    resumeData: ClaudeSDKAgentResumeData,
    options?: ClaudeSDKAgentRunOptions<OUTPUT>,
  ): Promise<MastraModelOutput<OUTPUT>> {
    const data = validateClaudeResumeData(resumeData);
    return this.stream(data.message, createClaudeResumeRunOptions(data, options));
  }
}

function validateClaudeResumeData(resumeData: ClaudeSDKAgentResumeData): ClaudeSDKAgentResumeData {
  if (!isRecord(resumeData) || !('message' in resumeData)) {
    throw new Error('ClaudeSDKAgent resumeData must include a message.');
  }

  const hasSessionId = 'sessionId' in resumeData;
  const hasContinue = 'continue' in resumeData;

  if (hasSessionId && hasContinue) {
    throw new Error('ClaudeSDKAgent resumeData must include either sessionId or continue: true, not both.');
  }

  if (hasSessionId) {
    if (typeof resumeData.sessionId !== 'string') {
      throw new Error('ClaudeSDKAgent resumeData.sessionId must be a string.');
    }
    return resumeData;
  }

  if (hasContinue) {
    if (resumeData.continue !== true) {
      throw new Error('ClaudeSDKAgent resumeData.continue must be true when provided.');
    }
    return resumeData;
  }

  throw new Error('ClaudeSDKAgent resumeData must include sessionId or continue: true.');
}

function createClaudeResumeRunOptions<OUTPUT>(
  resumeData: ClaudeSDKAgentResumeData,
  options?: ClaudeSDKAgentRunOptions<OUTPUT>,
): ClaudeSDKAgentRunOptions<OUTPUT> {
  const sdkOptions: ClaudeSDKOptions = { ...options?.sdkOptions };

  if ('sessionId' in resumeData && typeof resumeData.sessionId === 'string') {
    sdkOptions.resume = resumeData.sessionId;
    if (resumeData.forkSession !== undefined) {
      sdkOptions.forkSession = resumeData.forkSession;
    }
    if (resumeData.resumeSessionAt !== undefined) {
      sdkOptions.resumeSessionAt = resumeData.resumeSessionAt;
    }
  } else {
    sdkOptions.continue = true;
  }

  return {
    ...options,
    sdkOptions,
  };
}

async function runClaudeGenerate<OUTPUT>(
  prompt: string,
  options: ClaudeAgentOptions,
  telemetry: SDKAgentTelemetry<OUTPUT>,
  runOptions?: ClaudeSDKAgentRunOptions<OUTPUT>,
): Promise<SDKModelGenerateResult> {
  let text = '';
  let structuredOutputValue: unknown;
  const usage = createClaudeUsageCollector();

  for await (const message of observeClaudeMessages(
    runClaude(prompt, options, runOptions?.abortSignal ?? runOptions?.signal, runOptions),
    telemetry,
  )) {
    usage.record(message);
    if (message.type === 'result') {
      if (message.subtype !== 'success') {
        throw new Error(message.errors.join('\n') || `Claude Agent SDK failed with ${message.subtype}`);
      }

      text = message.result;
      structuredOutputValue = getClaudeStructuredOutput(message);
    }
  }

  const totals = usage.totals();
  const object = await getStructuredOutputFromValue(
    structuredOutputValue === undefined ? text : structuredOutputValue,
    runOptions?.structuredOutput,
  );

  return {
    content: [{ type: 'text', text }],
    finishReason: { unified: 'stop', raw: 'stop' },
    usage: usage.toV3Usage(),
    response: {
      id: randomUUID(),
      modelId: getModelId(options),
      timestamp: new Date(),
    },
    providerMetadata: getClaudeProviderMetadata(options, totals),
    costContext: getClaudeCostContext(options, totals),
    object,
  };
}

function runClaudeAsMastraStream<OUTPUT>(
  prompt: string,
  options: ClaudeAgentOptions,
  runId: string,
  telemetry: SDKAgentTelemetry<OUTPUT>,
  runOptions?: ClaudeSDKAgentRunOptions<OUTPUT>,
): ReadableStream<ChunkType> {
  return new ReadableStream<ChunkType>({
    start: async controller => {
      const textId = randomUUID();
      const responseId = randomUUID();
      const modelId = getModelId(options);
      const usage = createClaudeUsageCollector();
      let text = '';
      let structuredOutputValue: unknown;
      let sawDelta = false;

      try {
        enqueueStartChunks(controller, {
          runId,
          prompt,
          textId,
          responseId,
          modelId,
          providerMetadata: getClaudeProviderMetadata(options, usage.totals()),
        });

        for await (const message of observeClaudeMessages(
          runClaude(prompt, options, runOptions?.abortSignal ?? runOptions?.signal, runOptions),
          telemetry,
        )) {
          usage.record(message);
          const delta = getTextDelta(message);
          if (delta) {
            sawDelta = true;
            text += delta;
            enqueueTextDelta(controller, runId, textId, delta);
          }

          if (message.type === 'result') {
            if (message.subtype !== 'success') {
              throw new Error(message.errors.join('\n') || `Claude Agent SDK failed with ${message.subtype}`);
            }

            if (!sawDelta && message.result) {
              text += message.result;
              enqueueTextDelta(controller, runId, textId, message.result);
            }
            structuredOutputValue = getClaudeStructuredOutput(message);
          }
        }

        const totals = usage.totals();
        const providerMetadata = getClaudeProviderMetadata(options, totals);
        enqueueFinishChunks(controller, {
          runId,
          prompt,
          textId,
          text,
          responseId,
          modelId,
          usage: usage.toLanguageModelUsage(),
          providerMetadata,
          costContext: getClaudeCostContext(options, totals),
          object: await getStructuredOutputFromValue(
            structuredOutputValue === undefined ? text : structuredOutputValue,
            runOptions?.structuredOutput,
          ),
        });
        controller.close();
      } catch (error) {
        controller.enqueue({
          type: 'error',
          runId,
          from: ChunkFrom.AGENT,
          payload: { error },
        });
        controller.close();
      }
    },
  });
}

function runClaude<OUTPUT>(
  prompt: string,
  options: ClaudeAgentOptions,
  signal?: AbortSignal,
  runOptions?: ClaudeSDKAgentRunOptions<OUTPUT>,
): AsyncIterable<SDKMessage> {
  const abortController = createAbortController(signal);
  const queryOptions: ClaudeSDKOptions = {
    ...options.sdkOptions,
    ...runOptions?.sdkOptions,
  };
  const outputSchema = getStructuredOutputSchema(runOptions?.structuredOutput);
  if (outputSchema) {
    queryOptions.outputFormat = {
      type: 'json_schema',
      schema: outputSchema,
    };
  }
  if (abortController) {
    queryOptions.abortController = abortController;
  }

  return query({
    prompt,
    options: queryOptions as Parameters<typeof query>[0]['options'],
  }) as AsyncIterable<SDKMessage>;
}

function getClaudeStructuredOutput(message: SDKMessage): unknown {
  if (message.type !== 'result') {
    return undefined;
  }

  return (message as { structured_output?: unknown }).structured_output;
}

async function* observeClaudeMessages<OUTPUT>(
  messages: AsyncIterable<SDKMessage>,
  telemetry: SDKAgentTelemetry<OUTPUT>,
): AsyncIterable<SDKMessage> {
  for await (const message of messages) {
    recordClaudeToolTelemetry(message, telemetry);
    yield message;
  }
}

function recordClaudeToolTelemetry<OUTPUT>(message: SDKMessage, telemetry: SDKAgentTelemetry<OUTPUT>): void {
  for (const toolCall of getClaudeToolCalls(message)) {
    telemetry.startToolCall(toolCall);
  }

  for (const toolResult of getClaudeToolResults(message)) {
    telemetry.endToolCall(toolResult);
  }
}

function getClaudeToolCalls(message: SDKMessage): Array<{ toolCallId: string; toolName: string; input?: unknown }> {
  if (message.type !== 'assistant') {
    return [];
  }

  return getContentBlocks(message.message)
    .filter(isRecord)
    .filter(block => block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string')
    .map(block => ({
      toolCallId: block.id as string,
      toolName: block.name as string,
      input: block.input,
    }));
}

function getClaudeToolResults(message: SDKMessage): Array<{ toolCallId: string; output?: unknown; isError?: boolean }> {
  if (message.type !== 'user') {
    return [];
  }

  return getContentBlocks(message.message)
    .filter(isRecord)
    .filter(block => block.type === 'tool_result' && typeof block.tool_use_id === 'string')
    .map(block => ({
      toolCallId: block.tool_use_id as string,
      output: block.content,
      isError: block.is_error === true,
    }));
}

function getContentBlocks(message: unknown): unknown[] {
  if (!isRecord(message)) {
    return [];
  }

  return Array.isArray(message.content) ? message.content : [];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object';
}

function createAbortController(signal: AbortSignal | undefined): AbortController | undefined {
  if (!signal) {
    return undefined;
  }

  const controller = new AbortController();
  if (signal.aborted) {
    controller.abort(signal.reason);
    return controller;
  }

  signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true });
  return controller;
}

function getModelId(options: ClaudeAgentOptions): string {
  return options.sdkOptions?.model ?? MODEL_ID;
}

function createClaudeUsageCollector() {
  const assistantUsageById = new Map<string, ClaudeUsageTotals>();
  let resultUsage: ClaudeUsageTotals = {};

  return {
    record(message: SDKMessage) {
      if (message.type === 'assistant') {
        assistantUsageById.set(message.message.id, usageFromClaudeMessage(message.message.usage));
        return;
      }

      if (message.type === 'result') {
        resultUsage = {
          ...usageFromClaudeMessage(message.usage),
          totalCostUsd: message.total_cost_usd,
          modelUsage: message.modelUsage,
        };
      }
    },
    totals(): ClaudeUsageTotals {
      const assistantUsage = getAssistantUsageTotals(assistantUsageById);

      if (hasAnyUsage(resultUsage)) {
        return {
          ...resultUsage,
          inputTokens: resultUsage.inputTokens ?? assistantUsage.inputTokens,
          outputTokens: resultUsage.outputTokens ?? assistantUsage.outputTokens,
          cacheReadInputTokens: resultUsage.cacheReadInputTokens ?? assistantUsage.cacheReadInputTokens,
          cacheCreationInputTokens: resultUsage.cacheCreationInputTokens ?? assistantUsage.cacheCreationInputTokens,
        };
      }

      return assistantUsage;
    },
    toV3Usage(): V3Usage {
      return toV3Usage(this.totals());
    },
    toLanguageModelUsage(): LanguageModelUsage {
      return toLanguageModelUsage(toV3Usage(this.totals()));
    },
  };
}

function getAssistantUsageTotals(assistantUsageById: Map<string, ClaudeUsageTotals>): ClaudeUsageTotals {
  return [...assistantUsageById.values()].reduce<ClaudeUsageTotals>((totals, item) => {
    totals.inputTokens = addOptional(totals.inputTokens, item.inputTokens);
    totals.outputTokens = addOptional(totals.outputTokens, item.outputTokens);
    totals.cacheReadInputTokens = addOptional(totals.cacheReadInputTokens, item.cacheReadInputTokens);
    totals.cacheCreationInputTokens = addOptional(totals.cacheCreationInputTokens, item.cacheCreationInputTokens);
    return totals;
  }, {});
}

function usageFromClaudeMessage(usage: unknown): ClaudeUsageTotals {
  if (!usage || typeof usage !== 'object') {
    return {};
  }

  const record = usage as Record<string, unknown>;
  return {
    inputTokens: getTokenTotal(record.input_tokens),
    outputTokens: getTokenTotal(record.output_tokens),
    cacheReadInputTokens: getTokenTotal(record.cache_read_input_tokens),
    cacheCreationInputTokens: getTokenTotal(record.cache_creation_input_tokens),
  };
}

function hasAnyUsage(usage: ClaudeUsageTotals): boolean {
  return (
    usage.inputTokens !== undefined ||
    usage.outputTokens !== undefined ||
    usage.cacheReadInputTokens !== undefined ||
    usage.cacheCreationInputTokens !== undefined ||
    usage.totalCostUsd !== undefined
  );
}

function addOptional(left: number | undefined, right: number | undefined): number | undefined {
  if (left === undefined) {
    return right;
  }
  if (right === undefined) {
    return left;
  }

  return left + right;
}

function toV3Usage(usage: ClaudeUsageTotals): V3Usage {
  const noCache = usage.inputTokens;
  const cacheRead = usage.cacheReadInputTokens;
  const cacheWrite = usage.cacheCreationInputTokens;
  const totalInputTokens = sumDefined(noCache, cacheRead, cacheWrite);
  const outputTokens = usage.outputTokens;

  return {
    inputTokens: {
      total: totalInputTokens,
      noCache,
      cacheRead,
      cacheWrite,
    },
    outputTokens: {
      total: outputTokens,
      text: outputTokens,
    },
  };
}

function getClaudeProviderMetadata(options: ClaudeAgentOptions, usage?: ClaudeUsageTotals): ProviderMetadata {
  const queryOptions = options.sdkOptions;

  return createProviderMetadata('claude', {
    totalCostUsd: usage?.totalCostUsd,
    model: getModelId(options),
    cwd: queryOptions?.cwd,
    permissionMode: queryOptions?.permissionMode,
    maxTurns: queryOptions?.maxTurns,
    allowedTools: queryOptions?.allowedTools,
    disallowedTools: queryOptions?.disallowedTools,
    usage,
  });
}

function getClaudeCostContext(options: ClaudeAgentOptions, usage?: ClaudeUsageTotals): CostContext | undefined {
  if (typeof usage?.totalCostUsd !== 'number') {
    return undefined;
  }

  return {
    provider: 'anthropic',
    model: getModelId(options),
    estimatedCost: usage.totalCostUsd,
    costUnit: 'USD',
    costMetadata: {
      source: 'sdk_estimate',
      sdkProvider: PROVIDER,
      sdkCostField: 'total_cost_usd',
      scope: 'query_total',
      modelUsage: usage.modelUsage,
    },
  };
}

function getTextDelta(message: SDKMessage): string {
  if (message.type !== 'stream_event') {
    return '';
  }

  const event = message.event as {
    type?: string;
    delta?: {
      type?: string;
      text?: string;
    };
  };

  if (event.type === 'content_block_delta' && event.delta?.type === 'text_delta') {
    return event.delta.text ?? '';
  }

  return '';
}

function getTokenTotal(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
}

// =============================================================================
// AgentController bridge
// =============================================================================

/** Name of the in-process MCP server the controller's tools are bridged through. */
const MCP_SERVER = 'mastra';
/** Controller tools a Claude Code session does not need: it has its own subagents and todo list. */
const SKIPPED_TOOLS = new Set(['subagent', 'task_write', 'task_update', 'task_complete', 'task_check']);
const WORKDIR_TIMEOUT_MS = 120_000;

export type ClaudeCodeControllerAgentOptions = ClaudeAgentOptions & {
  /**
   * Tools bridged into every session next to the controller built-ins (`ask_user`, `submit_plan`);
   * same shape as `AgentControllerConfig.tools`. The agent owns them rather than reading
   * `options.toolsets`, because a signal replayed from the thread queue carries neither toolsets
   * nor request context.
   */
  tools?: ToolsInput | ((ctx: { requestContext: RequestContext }) => Promise<ToolsInput | undefined> | ToolsInput | undefined);
  /**
   * Working directory for a thread, tried before the controller state's `projectPath` and
   * `sdkOptions.cwd`. A host whose session workspace is keyed by something other than the thread
   * id resolves it here.
   */
  resolveCwd?: (threadId: string) => string | undefined | Promise<string | undefined>;
};

type ToolLike = {
  description?: string;
  inputSchema?: unknown;
  execute?: (input: unknown, context: unknown) => Promise<unknown>;
};

type Invocation = { toolCallId: string; toolName: string; args: unknown; result?: unknown };

/** Thrown by a bridged tool's `suspend()` so the tool's own `execute` re-runs with the resume data. */
class ResumeSignal extends Error {
  constructor(readonly resumeData: unknown) {
    super('resume');
  }
}

function signalText(messages: unknown): string {
  const signal = messages as { __isCreatedSignal?: true; toLLMMessage?: () => { content: unknown } };
  return signal?.__isCreatedSignal && signal.toLLMMessage ? promptToText(signal.toLLMMessage()) : promptToText(messages);
}

function zodShape(schema: unknown): Record<string, unknown> {
  const s = schema as { shape?: Record<string, unknown>; in?: { shape?: Record<string, unknown> } } | undefined;
  return s?.shape ?? s?.in?.shape ?? {};
}

function toolResultText(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value ?? null);
}

async function waitForDirectory(path: string, timeoutMs: number): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!existsSync(path)) {
    if (Date.now() > deadline) {
      throw new Error(`ClaudeCodeControllerAgent: working directory ${path} did not appear within ${timeoutMs}ms`);
    }
    await sleep(250);
  }
}

/**
 * A `ClaudeSDKAgent` that satisfies `AgentController`, so a controller-driven product (Mastra Code,
 * Factory) runs its sessions on the Claude Code binary instead of the built-in code agent.
 *
 * What the controller expects of an agent, and how this supplies it:
 *
 * - **Run bookkeeping.** `Agent.stream` registers its run with the thread runtime, which pumps the
 *   output, observes it as finished and drains signals queued meanwhile. An override that drives an
 *   external harness must do the same or the thread stays reserved and later messages never arrive.
 * - **Tools.** Controller toolsets are bridged into one in-process MCP server, so `ask_user`,
 *   `submit_plan` and the host's own tools are callable from the session.
 * - **Suspension.** A bridged tool's `suspend()` emits `tool-call-suspended` and parks; the
 *   controller's `sendStreamResume` resolves it and the tool's `execute` re-runs with the resume data.
 * - **Thread content.** Native tool activity is mirrored as chunks and each turn is written back to
 *   memory, so the thread shows what the session did.
 * - **`untilIdle`.** Signals queued during a turn become the next input of the same run, in the same
 *   Claude session.
 *
 * One Claude session per Mastra thread: a new session id on the first turn, `resume` after.
 */
export class ClaudeCodeControllerAgent extends ClaudeSDKAgent {
  readonly #options: ClaudeCodeControllerAgentOptions;
  readonly #sessionByThread = new Map<string, string>();
  readonly #pendingResume = new Map<string, (resumeData: unknown) => void>();
  readonly #runByToolCall = new Map<string, string>();

  constructor(options: ClaudeCodeControllerAgentOptions) {
    super(options);
    this.#options = options;
  }

  async #resolveTools(requestContext: RequestContext): Promise<Map<string, ToolLike>> {
    let own: ToolsInput | undefined;
    try {
      own = typeof this.#options.tools === 'function' ? await this.#options.tools({ requestContext }) : this.#options.tools;
    } catch (error) {
      this.logger.warn(`ClaudeCodeControllerAgent: tools resolver failed`, { error });
    }
    const bridged = new Map<string, ToolLike>();
    for (const [name, tool] of Object.entries({ ask_user: askUserTool, submit_plan: submitPlanTool, ...(own ?? {}) })) {
      const candidate = tool as ToolLike;
      if (!SKIPPED_TOOLS.has(name) && candidate?.execute) bridged.set(name, candidate);
    }
    return bridged;
  }

  async #resolveCwd(threadId: string | undefined, requestContext: RequestContext): Promise<string | undefined> {
    const controller = requestContext.get('controller') as { getState?: () => { projectPath?: string } | undefined } | undefined;
    return (
      (threadId ? await this.#options.resolveCwd?.(threadId) : undefined) ||
      controller?.getState?.()?.projectPath ||
      this.#options.sdkOptions?.cwd
    );
  }

  override async sendStreamResume<OUTPUT = undefined>(
    options: SendAgentStreamResumeOptions<OUTPUT>,
  ): Promise<SendAgentStreamResumeResult> {
    const { toolCallId, resumeData } = options;
    const resolve = toolCallId ? this.#pendingResume.get(toolCallId) : undefined;
    if (!toolCallId || !resolve) throw new Error(`ClaudeCodeControllerAgent: no suspended tool call ${toolCallId}`);
    this.#pendingResume.delete(toolCallId);
    resolve(resumeData);
    return { accepted: true, runId: this.#runByToolCall.get(toolCallId) ?? options.runId, toolCallId };
  }

  override async stream<OUTPUT = undefined>(
    messages: MessageListInput,
    options?: ClaudeSDKAgentRunOptions<OUTPUT>,
  ): Promise<MastraModelOutput<OUTPUT>> {
    const runId = options?.runId ?? randomUUID();
    const prompt = signalText(messages);
    const modelId = getModelId(this.#options);
    const memoryOpt = (options as { memory?: { thread?: string | { id: string }; resource?: string } })?.memory;
    const threadId = typeof memoryOpt?.thread === 'string' ? memoryOpt.thread : memoryOpt?.thread?.id;
    const resourceId = memoryOpt?.resource;

    // The thread runtime observes a run (pump, completion, queued follow-ups) only once it is
    // registered; `Agent.stream` does this around its own loop.
    const pubsub = this.getPubSub();
    await agentThreadStreamRuntime.waitForCrossAgentThreadRun(
      this as never,
      { memory: memoryOpt as never, requestContext: options?.requestContext, runId },
      pubsub,
    );
    const prepared = agentThreadStreamRuntime.prepareRunOptions(
      { ...options, runId, requestContext: options?.requestContext ?? new RequestContext() } as never,
      pubsub,
    ) as ClaudeSDKAgentRunOptions<OUTPUT> & { requestContext: RequestContext; abortSignal?: AbortSignal };
    const requestContext = prepared.requestContext;

    const cwd = await this.#resolveCwd(threadId, requestContext);
    const bridged = await this.#resolveTools(requestContext);
    const telemetry = createSDKAgentTelemetry({
      agentId: this.id,
      agentName: this.name,
      provider: PROVIDER,
      modelId,
      messages,
      prompt,
      runId,
      streaming: true,
      method: 'stream',
      requestContext,
      instructions: undefined,
      maxSteps: options?.maxSteps,
      tracingOptions: options?.tracingOptions,
      tracingContext: options?.tracingContext,
      onFinish: options?.onFinish,
      onStepFinish: options?.onStepFinish,
      mastra: undefined,
    });
    this.logger.debug('ClaudeCodeControllerAgent: run starting', {
      runId,
      threadId,
      cwd,
      tools: [...bridged.keys()],
    });

    const stream = new ReadableStream<ChunkType>({
      start: async controller => {
        const textId = randomUUID();
        const responseId = randomUUID();
        const usage = createClaudeUsageCollector();
        const toolNames = new Map<string, string>();
        const invocations: Invocation[] = [];
        let text = '';

        const toolCall = (toolCallId: string, toolName: string, args: unknown) => {
          toolNames.set(toolCallId, toolName);
          invocations.push({ toolCallId, toolName, args });
          controller.enqueue({ type: 'tool-call', runId, from: ChunkFrom.AGENT, payload: { toolCallId, toolName, args } } as ChunkType);
        };
        const toolResult = (toolCallId: string, result: unknown, isError = false) => {
          const toolName = toolNames.get(toolCallId) ?? '';
          const invocation = invocations.find(i => i.toolCallId === toolCallId);
          if (invocation) invocation.result = result;
          controller.enqueue({
            type: 'tool-result',
            runId,
            from: ChunkFrom.AGENT,
            payload: { toolCallId, toolName, result, isError },
          } as ChunkType);
        };

        const mcpTools = [...bridged].map(([name, tool]) =>
          sdkTool(name, tool.description ?? name, zodShape(tool.inputSchema) as any, async (args: unknown) => {
            const toolCallId = randomUUID();
            this.#runByToolCall.set(toolCallId, runId);
            toolCall(toolCallId, name, args);
            const context = (resumeData?: unknown) => ({
              requestContext,
              agent: {
                agentId: this.id,
                toolCallId,
                messages: [],
                threadId,
                resourceId,
                resumeData,
                suspend: async (suspendPayload: unknown) => {
                  controller.enqueue({
                    type: 'tool-call-suspended',
                    runId,
                    from: ChunkFrom.AGENT,
                    payload: { toolCallId, toolName: name, args, suspendPayload, resumeSchema: '{}' },
                  } as ChunkType);
                  this.logger.debug('ClaudeCodeControllerAgent: tool suspended', { runId, tool: name, toolCallId });
                  throw new ResumeSignal(await new Promise<unknown>(resolve => this.#pendingResume.set(toolCallId, resolve)));
                },
              },
            });
            try {
              let result: unknown;
              try {
                result = await tool.execute!(args, context());
              } catch (error) {
                if (!(error instanceof ResumeSignal)) throw error;
                result = await tool.execute!(args, context(error.resumeData));
              }
              toolResult(toolCallId, result);
              return { content: [{ type: 'text', text: toolResultText(result) }] };
            } catch (error) {
              const message = error instanceof Error ? error.message : String(error);
              toolResult(toolCallId, message, true);
              return { content: [{ type: 'text', text: message }], isError: true };
            }
          }),
        );

        const newSessionId = randomUUID();
        const sdkOptions = (): ClaudeSDKOptions => {
          const claudeSessionId = threadId ? this.#sessionByThread.get(threadId) : undefined;
          return {
            ...this.#options.sdkOptions,
            ...options?.sdkOptions,
            cwd,
            includePartialMessages: true,
            mcpServers: {
              ...this.#options.sdkOptions?.mcpServers,
              [MCP_SERVER]: createSdkMcpServer({ name: MCP_SERVER, tools: mcpTools }),
            },
            allowedTools: [
              ...(this.#options.sdkOptions?.allowedTools ?? []),
              ...[...bridged.keys()].map(name => `mcp__${MCP_SERVER}__${name}`),
            ],
            ...(claudeSessionId ? { resume: claudeSessionId } : { sessionId: newSessionId }),
          };
        };

        /** One Claude query: native tool activity mirrored as chunks, the turn written back to the thread. */
        const turn = async (turnPrompt: string) => {
          let turnText = '';
          const turnStart = invocations.length;
          for await (const message of observeClaudeMessages(
            runClaude(turnPrompt, { ...this.#options, sdkOptions: sdkOptions() }, prepared.abortSignal ?? options?.signal, undefined),
            telemetry,
          )) {
            usage.record(message);
            if (message.type === 'system' && message.subtype === 'init') {
              if (threadId) this.#sessionByThread.set(threadId, message.session_id);
              this.logger.debug('ClaudeCodeControllerAgent: session init', {
                runId,
                sessionId: message.session_id,
                model: message.model,
                apiKeySource: (message as { apiKeySource?: string }).apiKeySource,
                cwd: message.cwd,
              });
            }
            const delta = getTextDelta(message);
            if (delta) {
              turnText += delta;
              enqueueTextDelta(controller, runId, textId, delta);
            }
            if (message.type === 'assistant') {
              for (const call of getClaudeToolCalls(message)) {
                if (call.toolName.startsWith(`mcp__${MCP_SERVER}__`)) continue; // already mirrored by the bridge
                toolCall(call.toolCallId, call.toolName, call.input);
              }
            }
            if (message.type === 'user') {
              for (const result of getClaudeToolResults(message)) {
                if (toolNames.has(result.toolCallId)) toolResult(result.toolCallId, result.output, result.isError);
              }
            }
            if (message.type === 'result') {
              if (message.subtype !== 'success') {
                throw new Error(message.errors.join('\n') || `Claude Agent SDK failed with ${message.subtype}`);
              }
              if (!turnText && message.result) {
                turnText = message.result;
                enqueueTextDelta(controller, runId, textId, message.result);
              }
            }
          }
          text += turnText;
          await this.#writeBack({ requestContext, threadId, resourceId, text: turnText, invocations: invocations.slice(turnStart) });
        };

        try {
          // A host may materialize the session's checkout concurrently with the first message;
          // spawning the binary before its cwd exists fails the launch.
          if (cwd) await waitForDirectory(cwd, WORKDIR_TIMEOUT_MS);
          enqueueStartChunks(controller, {
            runId,
            prompt,
            textId,
            responseId,
            modelId,
            providerMetadata: getClaudeProviderMetadata(this.#options, usage.totals()),
          });
          let turnPrompt = prompt;
          for (;;) {
            await turn(turnPrompt);
            // `untilIdle`: signals queued on this run while it worked become the next input of the
            // same run and the same Claude session, persisted under their own ids first.
            const queued = agentThreadStreamRuntime.drainPendingSignals(runId, pubsub);
            if (queued.length === 0) break;
            const memory = await this.getMemory({ requestContext });
            if (memory && threadId) {
              await memory.saveMessages({ messages: queued.map(s => s.toDBMessage({ resourceId: resourceId ?? '', threadId })) });
            }
            for (const signal of queued) controller.enqueue({ ...signal.toDataPart(), runId } as unknown as ChunkType);
            turnPrompt = queued.map(s => promptToText(s.toLLMMessage())).join('\n\n');
            this.logger.debug('ClaudeCodeControllerAgent: drained queued signals', { runId, count: queued.length });
          }
          const totals = usage.totals();
          enqueueFinishChunks(controller, {
            runId,
            prompt,
            textId,
            text,
            responseId,
            modelId,
            usage: usage.toLanguageModelUsage(),
            providerMetadata: getClaudeProviderMetadata(this.#options, totals),
            costContext: getClaudeCostContext(this.#options, totals),
            object: undefined,
          });
          controller.close();
        } catch (error) {
          this.logger.error('ClaudeCodeControllerAgent: run failed', { runId, error });
          controller.enqueue({ type: 'error', runId, from: ChunkFrom.AGENT, payload: { error } } as ChunkType);
          controller.close();
        }
      },
    });

    const output = createMastraOutput<OUTPUT>({
      messages,
      runId,
      modelId,
      provider: PROVIDER,
      stream: telemetry.wrapStream(stream),
      options: { ...telemetry.outputOptions(), structuredOutput: options?.structuredOutput as never },
    });
    try {
      await agentThreadStreamRuntime.registerRun(this as never, output, prepared as never, pubsub);
    } catch (error) {
      agentThreadStreamRuntime.releaseThreadRunReservation(runId, pubsub, { agent: this as never, streamOptions: prepared as never });
      throw error;
    }
    return output;
  }

  async #writeBack({
    requestContext,
    threadId,
    resourceId,
    text,
    invocations,
  }: {
    requestContext: RequestContext;
    threadId?: string;
    resourceId?: string;
    text: string;
    invocations: Invocation[];
  }): Promise<void> {
    if (!threadId) return;
    const memory = await this.getMemory({ requestContext });
    if (!memory) return;
    const message = {
      id: randomUUID(),
      role: 'assistant',
      createdAt: new Date(),
      threadId,
      resourceId,
      type: 'text',
      content: {
        format: 2,
        parts: [
          ...invocations.map(i => ({
            type: 'tool-invocation' as const,
            toolInvocation: {
              state: 'result' as const,
              toolCallId: i.toolCallId,
              toolName: i.toolName,
              args: i.args,
              result: i.result ?? null,
            },
          })),
          ...(text ? [{ type: 'text' as const, text }] : []),
        ],
      },
    } as MastraDBMessage;
    await memory.saveMessages({ messages: [message] });
  }
}
