"""The low-level agent loop.

Python port of `packages/agent/src/agent-loop.ts`. The loop works with
``AgentMessage`` values throughout and converts to provider ``Message`` values
only at the LLM call boundary.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from pi_ai import (
    AssistantMessage,
    Context,
    EventStream,
    TextContent,
    ToolResultMessage,
    now_ms,
    validate_tool_arguments,
)
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.tasks import spawn

from .stream_fn import get_default_stream_fn
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]

TRUNCATED_TOOL_CALL_MESSAGE = (
    'Tool call "{name}" was not executed: the response hit the output token limit, '
    "so its arguments may be truncated. Re-issue the tool call with complete arguments."
)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    return EventStream(
        lambda event: event.type == "agent_end",
        lambda event: event.messages if event.type == "agent_end" else [],
    )


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Start an agent loop with new prompt messages."""
    stream = _create_agent_stream()

    async def run() -> None:
        messages = await run_agent_loop(prompts, context, config, stream.push, signal, stream_fn)
        stream.end(messages)

    spawn(run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """Continue an agent loop from the current context without a new message.

    The last message in ``context`` must convert to a ``user`` or ``toolResult``
    message via ``convert_to_llm``; otherwise the provider rejects the request.
    """
    _assert_continuable(context)
    stream = _create_agent_stream()

    async def run() -> None:
        messages = await run_agent_loop_continue(context, config, stream.push, signal, stream_fn)
        stream.end(messages)

    spawn(run())
    return stream


def _assert_continuable(context: AgentContext) -> None:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))
    for prompt in prompts:
        await _maybe_await(emit(MessageStartEvent(message=prompt)))
        await _maybe_await(emit(MessageEndEvent(message=prompt)))

    await _run_loop(
        current_context,
        new_messages,
        config,
        signal,
        emit,
        stream_fn if stream_fn is not None else get_default_stream_fn(),
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortSignal | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    _assert_continuable(context)

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=context.messages,
        tools=context.tools,
    )

    await _maybe_await(emit(AgentStartEvent()))
    await _maybe_await(emit(TurnStartEvent()))

    await _run_loop(
        current_context,
        new_messages,
        config,
        signal,
        emit,
        stream_fn if stream_fn is not None else get_default_stream_fn(),
    )
    return new_messages


async def _run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> None:
    current_context = initial_context
    config = initial_config
    last_completed_turn: PrepareNextTurnContext | None = None
    # The user may have typed while the previous run was still finishing.
    pending_messages: list[AgentMessage] = await _call_optional_list(config.get_steering_messages)

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if last_completed_turn is not None:
                if config.prepare_next_turn is not None:
                    snapshot = await _maybe_await(config.prepare_next_turn(last_completed_turn))
                    if snapshot:
                        current_context = snapshot.context or current_context
                        reasoning = config.reasoning
                        if snapshot.thinking_level is not None:
                            reasoning = None if snapshot.thinking_level == "off" else snapshot.thinking_level
                        config = replace(
                            config,
                            model=snapshot.model or config.model,
                            reasoning=reasoning,
                        )
                # Preparation can be long-running (for example, compaction). Pick up steering
                # queued while it ran. Only poll again if the earlier poll returned nothing;
                # otherwise one-at-a-time mode would deliver two messages in this turn.
                if not pending_messages:
                    pending_messages = await _call_optional_list(config.get_steering_messages)
                await _maybe_await(emit(TurnStartEvent()))

            if pending_messages:
                for message in pending_messages:
                    await _maybe_await(emit(MessageStartEvent(message=message)))
                    await _maybe_await(emit(MessageEndEvent(message=message)))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            message = await _stream_assistant_response(current_context, config, signal, emit, stream_fn)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _maybe_await(emit(TurnEndEvent(message=message, tool_results=[])))
                await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                return

            tool_calls = [block for block in message.content if block.type == "toolCall"]

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                # A "length" stop means the output was cut off, so every tool call
                # may carry truncated arguments. Fail them all rather than execute
                # potentially broken calls.
                if message.stop_reason == "length":
                    batch = await _fail_tool_calls_from_truncated_message(tool_calls, emit)
                else:
                    batch = await _execute_tool_calls(current_context, message, config, signal, emit)
                tool_results.extend(batch.messages)
                has_more_tool_calls = not batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _maybe_await(emit(TurnEndEvent(message=message, tool_results=tool_results)))

            last_completed_turn = PrepareNextTurnContext(
                message=message,
                tool_results=tool_results,
                context=current_context,
                new_messages=new_messages,
            )

            if config.should_stop_after_turn is not None:
                should_stop = await _maybe_await(
                    config.should_stop_after_turn(
                        ShouldStopAfterTurnContext(
                            message=message,
                            tool_results=tool_results,
                            context=current_context,
                            new_messages=new_messages,
                        )
                    )
                )
                if should_stop:
                    await _maybe_await(emit(AgentEndEvent(messages=new_messages)))
                    return

            pending_messages = await _call_optional_list(config.get_steering_messages)

        follow_up_messages = await _call_optional_list(config.get_follow_up_messages)
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        break

    await _maybe_await(emit(AgentEndEvent(messages=new_messages)))


async def _call_optional_list(callback: Callable[[], Any] | None) -> list[AgentMessage]:
    if callback is None:
        return []
    result = await _maybe_await(callback())
    return list(result) if result else []


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
    stream_fn: StreamFn,
) -> AssistantMessage:
    """Stream one assistant response, converting agent messages for the LLM."""
    messages = context.messages
    if config.transform_context is not None:
        messages = await _maybe_await(config.transform_context(messages, signal))

    if config.convert_to_llm is None:
        raise ValueError("AgentLoopConfig.convert_to_llm is required")
    llm_messages = await _maybe_await(config.convert_to_llm(messages))

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=list(context.tools) if context.tools else None,
    )

    # Resolve the API key per call: OAuth tokens can expire during long tool phases.
    resolved_api_key = None
    if config.get_api_key is not None and config.model is not None:
        resolved_api_key = await _maybe_await(config.get_api_key(config.model.provider))
    resolved_api_key = resolved_api_key or config.api_key

    options = replace(config, api_key=resolved_api_key, signal=signal)
    response = await _maybe_await(stream_fn(config.model, llm_context, options))

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await _maybe_await(emit(MessageStartEvent(message=copy.copy(partial_message))))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _maybe_await(
                    emit(
                        MessageUpdateEvent(
                            message=copy.copy(partial_message),
                            assistant_message_event=event,
                        )
                    )
                )
        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                await _maybe_await(emit(MessageStartEvent(message=copy.copy(final_message))))
            await _maybe_await(emit(MessageEndEvent(message=final_message)))
            return final_message

    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _maybe_await(emit(MessageStartEvent(message=copy.copy(final_message))))
    await _maybe_await(emit(MessageEndEvent(message=final_message)))
    return final_message


@dataclass
class _ExecutedToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class _PreparedToolCall:
    tool_call: AgentToolCall
    tool: AgentTool
    args: Any


@dataclass
class _ImmediateToolCallOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass
class _ExecutedToolCallOutcome:
    result: AgentToolResult
    is_error: bool


@dataclass
class _FinalizedToolCallOutcome:
    tool_call: AgentToolCall
    result: AgentToolResult
    is_error: bool


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


def _should_terminate_tool_batch(finalized_calls: Sequence[_FinalizedToolCallOutcome]) -> bool:
    return bool(finalized_calls) and all(call.result.terminate is True for call in finalized_calls)


async def _fail_tool_calls_from_truncated_message(
    tool_calls: list[AgentToolCall], emit: AgentEventSink
) -> _ExecutedToolCallBatch:
    """Fail every tool call of a message truncated by the output token limit.

    Streamed arguments are finalized with a salvage parser, so a truncated
    message can yield arguments that parse and validate but are incomplete.
    """
    messages: list[ToolResultMessage] = []
    for tool_call in tool_calls:
        await _maybe_await(
            emit(ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments))
        )
        finalized = _FinalizedToolCallOutcome(
            tool_call=tool_call,
            result=_create_error_tool_result(TRUNCATED_TOOL_CALL_MESSAGE.format(name=tool_call.name)),
            is_error=True,
        )
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return _ExecutedToolCallBatch(messages=messages, terminate=False)


async def _execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    tool_calls = [block for block in assistant_message.content if block.type == "toolCall"]
    tools_by_name = {tool.name: tool for tool in (current_context.tools or [])}
    has_sequential_tool_call = any(
        tools_by_name.get(tool_call.name) is not None and tools_by_name[tool_call.name].execution_mode == "sequential"
        for tool_call in tool_calls
    )
    if config.tool_execution == "sequential" or has_sequential_tool_call:
        return await _execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )
    return await _execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, signal, emit)


async def _execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    finalized_calls: list[_FinalizedToolCallOutcome] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _maybe_await(
            emit(ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments))
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _ImmediateToolCallOutcome):
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call, result=preparation.result, is_error=preparation.is_error
            )
        else:
            executed = await _execute_prepared_tool_call(preparation, signal, emit)
            finalized = await _finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, signal
            )

        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)

        if signal is not None and signal.aborted:
            break

    return _ExecutedToolCallBatch(messages=messages, terminate=_should_terminate_tool_batch(finalized_calls))


async def _execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[AgentToolCall],
    config: AgentLoopConfig,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallBatch:
    entries: list[Any] = []

    for tool_call in tool_calls:
        await _maybe_await(
            emit(ToolExecutionStartEvent(tool_call_id=tool_call.id, tool_name=tool_call.name, args=tool_call.arguments))
        )

        preparation = await _prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if isinstance(preparation, _ImmediateToolCallOutcome):
            finalized = _FinalizedToolCallOutcome(
                tool_call=tool_call, result=preparation.result, is_error=preparation.is_error
            )
            await _emit_tool_execution_end(finalized, emit)
            entries.append(finalized)
            if signal is not None and signal.aborted:
                break
            continue

        # Collect a factory instead of starting the coroutine: no tool may run
        # until every before_tool_call gate in the batch has resolved, which is
        # what makes the abort check after each preparation meaningful.
        def make_runner(prepared: _PreparedToolCall = preparation):
            async def run_prepared() -> _FinalizedToolCallOutcome:
                if signal is not None and signal.aborted:
                    finalized_call = _FinalizedToolCallOutcome(
                        tool_call=prepared.tool_call,
                        result=_create_error_tool_result("Operation aborted"),
                        is_error=True,
                    )
                    await _emit_tool_execution_end(finalized_call, emit)
                    return finalized_call
                executed = await _execute_prepared_tool_call(prepared, signal, emit)
                finalized_call = await _finalize_executed_tool_call(
                    current_context, assistant_message, prepared, executed, config, signal
                )
                await _emit_tool_execution_end(finalized_call, emit)
                return finalized_call

            return run_prepared

        entries.append(make_runner())
        if signal is not None and signal.aborted:
            break

    # Start every prepared tool now, preserving assistant source order.
    started: list[Any] = [spawn(entry()) if callable(entry) else entry for entry in entries]
    ordered: list[_FinalizedToolCallOutcome] = []
    for entry in started:
        ordered.append(await entry if isinstance(entry, asyncio.Future) else entry)

    messages: list[ToolResultMessage] = []
    for finalized in ordered:
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return _ExecutedToolCallBatch(messages=messages, terminate=_should_terminate_tool_batch(ordered))


def _prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    if tool.prepare_arguments is None:
        return tool_call
    prepared_arguments = tool.prepare_arguments(tool_call.arguments)
    if prepared_arguments is tool_call.arguments:
        return tool_call
    prepared = copy.copy(tool_call)
    prepared.arguments = prepared_arguments
    return prepared


async def _prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
) -> _PreparedToolCall | _ImmediateToolCallOutcome:
    tool = next((t for t in (current_context.tools or []) if t.name == tool_call.name), None)
    if tool is None:
        return _ImmediateToolCallOutcome(
            result=_create_error_tool_result(f"Tool {tool_call.name} not found"), is_error=True
        )

    try:
        prepared_tool_call = _prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.before_tool_call is not None:
            before_result = await _maybe_await(
                config.before_tool_call(
                    BeforeToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=tool_call,
                        args=validated_args,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if signal is not None and signal.aborted:
                return _ImmediateToolCallOutcome(result=_create_error_tool_result("Operation aborted"), is_error=True)
            if before_result is not None and before_result.block:
                result = _create_error_tool_result(before_result.reason or "Tool execution was blocked")
                if before_result.terminate is True:
                    result.terminate = True
                return _ImmediateToolCallOutcome(result=result, is_error=True)
        if signal is not None and signal.aborted:
            return _ImmediateToolCallOutcome(result=_create_error_tool_result("Operation aborted"), is_error=True)
        return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)
    except Exception as error:
        return _ImmediateToolCallOutcome(result=_create_error_tool_result(str(error)), is_error=True)


async def _execute_prepared_tool_call(
    prepared: _PreparedToolCall,
    signal: AbortSignal | None,
    emit: AgentEventSink,
) -> _ExecutedToolCallOutcome:
    update_events: list[Any] = []
    accepting_updates = True

    def on_update(partial_result: AgentToolResult) -> None:
        if not accepting_updates:
            return
        # Start the sink eagerly so long-running tools stream progress instead
        # of having every update flushed after execute() returns.
        update_events.append(
            spawn(
                _maybe_await(
                    emit(
                        ToolExecutionUpdateEvent(
                            tool_call_id=prepared.tool_call.id,
                            tool_name=prepared.tool_call.name,
                            args=prepared.tool_call.arguments,
                            partial_result=partial_result,
                        )
                    )
                )
            )
        )

    try:
        if prepared.tool.execute is None:
            raise ValueError(f"Tool {prepared.tool.name} has no execute implementation")
        result = await prepared.tool.execute(prepared.tool_call.id, prepared.args, signal, on_update)
        accepting_updates = False
        await asyncio.gather(*update_events)
        return _ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as error:
        accepting_updates = False
        await asyncio.gather(*update_events)
        return _ExecutedToolCallOutcome(result=_create_error_tool_result(str(error)), is_error=True)


async def _finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    executed: _ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    signal: AbortSignal | None,
) -> _FinalizedToolCallOutcome:
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        try:
            after_result = await _maybe_await(
                config.after_tool_call(
                    AfterToolCallContext(
                        assistant_message=assistant_message,
                        tool_call=prepared.tool_call,
                        args=prepared.args,
                        result=result,
                        is_error=is_error,
                        context=current_context,
                    ),
                    signal,
                )
            )
            if after_result is not None:
                result = AgentToolResult(
                    content=after_result.content if after_result.content is not None else result.content,
                    details=after_result.details if after_result.details is not None else result.details,
                    usage=after_result.usage if after_result.usage is not None else result.usage,
                    added_tool_names=result.added_tool_names,
                    terminate=after_result.terminate if after_result.terminate is not None else result.terminate,
                )
                if after_result.is_error is not None:
                    is_error = after_result.is_error
        except Exception as error:
            result = _create_error_tool_result(str(error))
            is_error = True

    return _FinalizedToolCallOutcome(tool_call=prepared.tool_call, result=result, is_error=is_error)


async def _emit_tool_execution_end(finalized: _FinalizedToolCallOutcome, emit: AgentEventSink) -> None:
    await _maybe_await(
        emit(
            ToolExecutionEndEvent(
                tool_call_id=finalized.tool_call.id,
                tool_name=finalized.tool_call.name,
                result=finalized.result,
                is_error=finalized.is_error,
            )
        )
    )


def _create_tool_result_message(finalized: _FinalizedToolCallOutcome) -> ToolResultMessage:
    message = ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        # Untyped tools can return results without content; normalize so no None
        # enters session history or provider payloads.
        content=finalized.result.content or [],
        details=finalized.result.details,
        usage=finalized.result.usage,
        is_error=finalized.is_error,
        timestamp=now_ms(),
    )
    if finalized.result.added_tool_names:
        message.added_tool_names = finalized.result.added_tool_names
    return message


async def _emit_tool_result_message(tool_result_message: ToolResultMessage, emit: AgentEventSink) -> None:
    await _maybe_await(emit(MessageStartEvent(message=tool_result_message)))
    await _maybe_await(emit(MessageEndEvent(message=tool_result_message)))
