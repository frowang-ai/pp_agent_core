"""Python port of `packages/agent/test/agent-loop.test.ts`.

Every TypeScript `it(...)` in that file has a counterpart here. The port also
extends beyond it to cover loop behavior the TypeScript suite only exercises
through the `Agent` facade (steering/follow-up queues, `get_api_key`
resolution, tool-update ordering).

One structural difference recurs: TypeScript's `MockAssistantStream` pushes a
single `done` event per turn, while `agent_helpers.scripted_stream_fn` replays
the real delta protocol, so this port sees extra `message_update` events. Tests
that assert an exact event sequence filter those out at the assertion site.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from agent_helpers import (
    TEST_MODEL,
    error_response,
    make_assistant_message,
    scripted_stream_fn,
    text_response,
    tool_call_response,
)
from pi_ai import Cost, Model, TextContent, ToolCall, Usage, UserMessage, now_ms
from pi_ai.utils.abort import AbortController

from pi_agent import (
    AfterToolCallResult,
    AgentContext,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    agent_loop,
    agent_loop_continue,
    default_convert_to_llm,
    run_agent_loop,
)
from pi_agent.stream_fn import set_default_stream_fn


@dataclass
class _RoleMessage:
    """A message with an arbitrary role, like the duck-typed objects the TypeScript test uses."""

    role: str
    content: str
    timestamp: int = field(default_factory=lambda: now_ms())


def make_config(**overrides) -> AgentLoopConfig:
    defaults = dict(model=TEST_MODEL, convert_to_llm=default_convert_to_llm)
    defaults.update(overrides)
    return AgentLoopConfig(**defaults)


def echo_tool(name: str = "echo", **overrides) -> AgentTool:
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{params.get('value', '')}")], details={})

    defaults = dict(
        name=name,
        description="Echo a value",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        label=name,
        execute=execute,
    )
    defaults.update(overrides)
    return AgentTool(**defaults)


async def collect_events(stream):
    events = [event async for event in stream]
    return events, await stream.result()


async def test_single_turn_emits_full_lifecycle():
    stream_fn = scripted_stream_fn([text_response("hello")])
    events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(), make_config(), None, stream_fn)
    )

    assert [event.type for event in events] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_update",
        "message_update",
        "message_update",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[1].content[0].text == "hello"


async def test_tool_call_turn_executes_tool_and_continues():
    tool_call = ToolCall(id="c1", name="echo", arguments={"value": "x"})
    stream_fn = scripted_stream_fn([tool_call_response(tool_call), text_response("done")])
    context = AgentContext(tools=[echo_tool()])

    events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], context, make_config(), None, stream_fn)
    )

    types = [event.type for event in events]
    assert "tool_execution_start" in types
    assert "tool_execution_end" in types
    tool_results = [m for m in messages if m.role == "toolResult"]
    assert len(tool_results) == 1
    assert tool_results[0].content[0].text == "echo:x"
    assert tool_results[0].is_error is False
    assert messages[-1].content[0].text == "done"


async def test_missing_tool_produces_error_result():
    tool_call = ToolCall(id="c1", name="nope", arguments={})
    stream_fn = scripted_stream_fn([tool_call_response(tool_call), text_response("done")])

    _events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[]), make_config(), None, stream_fn)
    )

    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.is_error is True
    assert tool_result.content[0].text == "Tool nope not found"


async def test_invalid_tool_arguments_produce_error_result():
    tool_call = ToolCall(id="c1", name="echo", arguments={})
    stream_fn = scripted_stream_fn([tool_call_response(tool_call), text_response("done")])

    _events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[echo_tool()]), make_config(), None, stream_fn)
    )

    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.is_error is True
    assert "Validation failed" in tool_result.content[0].text


async def test_tool_exception_becomes_error_result():
    async def failing(tool_call_id, params, signal=None, on_update=None):
        raise RuntimeError("tool exploded")

    tool = echo_tool(execute=failing)
    tool_call = ToolCall(id="c1", name="echo", arguments={"value": "x"})
    stream_fn = scripted_stream_fn([tool_call_response(tool_call), text_response("done")])

    _events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.is_error is True
    assert tool_result.content[0].text == "tool exploded"


async def test_error_response_stops_the_loop():
    stream_fn = scripted_stream_fn([error_response("boom")])
    events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(), make_config(), None, stream_fn)
    )

    assert [event.type for event in events][-2:] == ["turn_end", "agent_end"]
    assert messages[-1].stop_reason == "error"
    assert messages[-1].error_message == "boom"


async def test_length_stop_reason_fails_all_tool_calls_without_executing():
    executed: list[str] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(tool_call_id)
        return AgentToolResult(content=[TextContent(text="ran")], details={})

    tool = echo_tool(execute=execute)
    truncated = make_assistant_message([ToolCall(id="c1", name="echo", arguments={"value": "x"})], stop_reason="length")
    stream_fn = scripted_stream_fn([truncated, text_response("done")])

    _events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    assert executed == []
    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.is_error is True
    assert "output token limit" in tool_result.content[0].text
    tool_end = next((event for event in _events if event.type == "tool_execution_end"), None)
    assert tool_end is not None
    assert tool_end.is_error is True
    assert "output token limit" in next(c.text for c in tool_end.result.content if c.type == "text")
    # The loop continues so the model can re-issue the tool call.
    assert len(stream_fn.calls) == 2
    assert messages[-1].role == "assistant"


async def test_before_tool_call_can_block_execution():
    executed: list[str] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(tool_call_id)
        return AgentToolResult(content=[TextContent(text="ran")], details={})

    async def before_tool_call(context, signal=None):
        return BeforeToolCallResult(block=True, reason="not allowed")

    tool = echo_tool(execute=execute)
    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[tool]),
            make_config(before_tool_call=before_tool_call),
            None,
            stream_fn,
        )
    )

    assert executed == []
    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.content[0].text == "not allowed"


async def test_blocked_tool_with_terminate_ends_the_batch():
    executed = {"ran": False}

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed["ran"] = True
        return AgentToolResult(content=[TextContent(text="should not execute")], details={})

    async def before_tool_call(context, signal=None):
        return BeforeToolCallResult(block=True, reason="Blocked by policy", terminate=True)

    stream_fn = scripted_stream_fn([tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"}))])

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool(execute=execute)]),
            make_config(before_tool_call=before_tool_call),
            None,
            stream_fn,
        )
    )

    # Only one provider call happened: the batch terminated instead of looping.
    assert executed["ran"] is False
    assert len(stream_fn.calls) == 1
    assert messages[-1].role == "toolResult"
    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.is_error is True
    assert TextContent(text="Blocked by policy") in tool_result.content


async def test_after_tool_call_overrides_result_fields():
    async def after_tool_call(context, signal=None):
        return AfterToolCallResult(content=[TextContent(text="overridden")], is_error=True)

    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool()]),
            make_config(after_tool_call=after_tool_call),
            None,
            stream_fn,
        )
    )

    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.content[0].text == "overridden"
    assert tool_result.is_error is True


async def test_after_tool_call_sees_the_tool_usage_and_can_replace_it():
    """The usage half of the TypeScript "should handle tool calls and results" case."""
    tool_usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1.0),
    )
    patched_tool_usage = Usage(
        input=5,
        output=6,
        cache_read=7,
        cache_write=8,
        total_tokens=26,
        cost=Cost(input=0.5, output=0.6, cache_read=0.7, cache_write=0.8, total=2.6),
    )
    executed: list[str] = []
    observed: list[Usage | None] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(params["value"])
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={"value": params["value"]},
            usage=tool_usage,
        )

    async def after_tool_call(context, signal=None):
        observed.append(context.result.usage)
        return AfterToolCallResult(usage=patched_tool_usage)

    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})), text_response("done")]
    )

    events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="echo something")],
            AgentContext(tools=[echo_tool(execute=execute)]),
            make_config(after_tool_call=after_tool_call),
            None,
            stream_fn,
        )
    )

    assert executed == ["hello"]
    tool_start = next((event for event in events if event.type == "tool_execution_start"), None)
    tool_end = next((event for event in events if event.type == "tool_execution_end"), None)
    assert tool_start is not None
    assert tool_end is not None
    assert tool_end.is_error is False
    assert observed == [tool_usage]
    tool_result = next(message for message in messages if message.role == "toolResult")
    assert tool_result.usage == patched_tool_usage


async def test_should_stop_after_turn_ends_the_run():
    stream_fn = scripted_stream_fn([text_response("first")])

    def should_stop(context):
        return True

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(),
            make_config(should_stop_after_turn=should_stop),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 1
    assert messages[-1].content[0].text == "first"


async def test_should_stop_after_turn_returning_false_lets_the_run_continue():
    """Guard for the falsy branch of `shouldStopAfterTurn`.

    The TypeScript stub is `async`, so the production callback resolves a Promise. If the loop ever
    dropped its `await` here, the un-awaited coroutine would be truthy and the run would stop after
    the first turn instead of continuing -- this is the only test that pins that direction.
    """
    calls: list[str] = []

    async def should_stop_after_turn(turn):
        calls.append(turn.message.role)
        return False

    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )
    events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool()]),
            make_config(should_stop_after_turn=should_stop_after_turn),
            None,
            stream_fn,
        )
    )

    # Invoked once per completed turn: the tool-call turn and the final text turn.
    assert calls == ["assistant", "assistant"]
    assert len(stream_fn.calls) == 2
    assert [m.role for m in messages] == ["user", "assistant", "toolResult", "assistant"]
    assert messages[-1].content[0].text == "done"
    assert [event.type for event in events][-2:] == ["turn_end", "agent_end"]


async def test_should_stop_after_turn_stops_the_run_and_sees_the_finished_turn():
    """Full assertion set from the TypeScript "stop after the current turn" case."""
    executed: list[str] = []
    polls = {"steering": 0, "follow_up": 0}
    callback: dict[str, list[str]] = {}

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={})

    async def get_steering_messages():
        polls["steering"] += 1
        return []

    async def get_follow_up_messages():
        polls["follow_up"] += 1
        return [UserMessage(content="follow up should stay queued")]

    # Async to match the TypeScript stub (`shouldStopAfterTurn: async ({...}) =>`) and the shape the
    # `Agent` facade really passes in (`async def wrapped_should_stop`). A sync stub would keep passing
    # even if the loop dropped its `await`, since an un-awaited coroutine is truthy.
    async def should_stop_after_turn(turn):
        assert turn.message.role == "assistant"
        callback["tool_result_ids"] = [result.tool_call_id for result in turn.tool_results]
        callback["context_roles"] = [message.role for message in turn.context.messages]
        return True

    stream_fn = scripted_stream_fn(
        [
            tool_call_response(ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})),
            text_response("should not run"),
        ]
    )

    events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="echo something")],
            AgentContext(tools=[echo_tool(execute=execute)]),
            make_config(
                get_steering_messages=get_steering_messages,
                get_follow_up_messages=get_follow_up_messages,
                should_stop_after_turn=should_stop_after_turn,
            ),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 1
    assert executed == ["hello"]
    assert polls["steering"] == 1
    assert polls["follow_up"] == 0
    assert callback["tool_result_ids"] == ["tool-1"]
    assert callback["context_roles"] == ["user", "assistant", "toolResult"]
    assert [message.role for message in messages] == ["user", "assistant", "toolResult"]
    # `message_update` has no TypeScript counterpart here: the mock stream in
    # `agent-loop.test.ts` pushes a single `done` event, while this port's
    # `scripted_stream_fn` replays the real delta protocol.
    assert [event.type for event in events if event.type != "message_update"] == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]


async def test_steering_messages_are_injected_before_the_next_turn():
    polls = {"count": 0}

    async def get_steering_messages():
        # The loop polls once before the first turn; make the message available
        # only afterwards so it is injected before the second turn.
        polls["count"] += 1
        if polls["count"] == 2:
            return [UserMessage(content="steer me")]
        return []

    stream_fn = scripted_stream_fn([text_response("first"), text_response("second")])

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(),
            make_config(get_steering_messages=get_steering_messages),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 2
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]


async def test_queued_messages_are_injected_after_all_tool_calls_complete():
    """Port of the TypeScript "should inject queued messages after all tool calls complete" case."""
    executed: list[str] = []
    queued_delivered = {"done": False}
    saw_interrupt_in_context = {"seen": False}

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"ok:{params['value']}")], details={})

    async def get_steering_messages():
        # Only offer the steering message once tool execution has started.
        if len(executed) >= 1 and not queued_delivered["done"]:
            queued_delivered["done"] = True
            return [UserMessage(content="interrupt")]
        return []

    message = make_assistant_message(
        [
            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
        ],
        stop_reason="toolUse",
    )
    scripted = scripted_stream_fn([message, text_response("done")])
    calls = {"count": 0}

    def stream_fn(model, context, options=None):
        if calls["count"] == 1:
            saw_interrupt_in_context["seen"] = any(
                m.role == "user" and m.content == "interrupt" for m in context.messages
            )
        calls["count"] += 1
        return scripted(model, context, options)

    events, _messages = await collect_events(
        agent_loop(
            [UserMessage(content="start")],
            AgentContext(tools=[echo_tool(execute=execute)]),
            make_config(tool_execution="sequential", get_steering_messages=get_steering_messages),
            None,
            stream_fn,
        )
    )

    assert executed == ["first", "second"]

    tool_ends = [event for event in events if event.type == "tool_execution_end"]
    assert len(tool_ends) == 2
    assert tool_ends[0].is_error is False
    assert tool_ends[1].is_error is False

    sequence: list[str] = []
    for event in events:
        if event.type != "message_start":
            continue
        if event.message.role == "toolResult":
            sequence.append(f"tool:{event.message.tool_call_id}")
        elif event.message.role == "user" and isinstance(event.message.content, str):
            sequence.append(event.message.content)
    assert "interrupt" in sequence
    assert sequence.index("tool:tool-1") < sequence.index("interrupt")
    assert sequence.index("tool:tool-2") < sequence.index("interrupt")
    assert saw_interrupt_in_context["seen"] is True


async def test_steering_messages_available_at_start_join_the_first_turn():
    steering = [UserMessage(content="steer me")]

    async def get_steering_messages():
        return [steering.pop(0)] if steering else []

    stream_fn = scripted_stream_fn([text_response("first")])

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(),
            make_config(get_steering_messages=get_steering_messages),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 1
    assert [m.role for m in messages] == ["user", "user", "assistant"]


async def test_follow_up_messages_restart_the_loop():
    follow_ups = [[UserMessage(content="follow up")]]

    async def get_follow_up_messages():
        return follow_ups.pop(0) if follow_ups else []

    stream_fn = scripted_stream_fn([text_response("first"), text_response("second")])

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(),
            make_config(get_follow_up_messages=get_follow_up_messages),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 2
    assert messages[-1].content[0].text == "second"


async def test_prepare_next_turn_can_swap_the_model():
    other_model = Model(id="other", provider="test", api="test-api", base_url="")

    async def prepare_next_turn(context):
        return AgentLoopTurnUpdate(model=other_model)

    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )

    await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool()]),
            make_config(prepare_next_turn=prepare_next_turn),
            None,
            stream_fn,
        )
    )

    assert stream_fn.calls[1]["model"] is other_model


async def test_prepare_next_turn_snapshot_is_used_before_continuing():
    """The TypeScript case swaps the whole context (system prompt) rather than the model."""
    prepared = {"done": False}

    # Async, mirroring `prepareNextTurn: async ({ context }) =>` in the TypeScript test.
    async def prepare_next_turn(turn):
        if prepared["done"]:
            return None
        prepared["done"] = True
        return AgentLoopTurnUpdate(
            context=AgentContext(
                system_prompt="second prompt",
                messages=list(turn.context.messages),
                tools=list(turn.context.tools),
            )
        )

    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})), text_response("done")]
    )

    await collect_events(
        agent_loop(
            [UserMessage(content="echo something")],
            AgentContext(system_prompt="first prompt", tools=[echo_tool()]),
            make_config(prepare_next_turn=prepare_next_turn),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 2
    assert stream_fn.calls[1]["context"].system_prompt == "second prompt"


async def test_prepare_next_turn_runs_once_per_continuing_turn():
    """`prepareNextTurn` fires at the start of each continued turn, not after every turn.

    Regression test for the port deviation where `prepare_next_turn` ran right
    after `turn_end`, before the `should_stop_after_turn` check -- so it also
    fired for turns the loop never continues from. agent-loop.ts keeps the last
    completed turn and only prepares at the top of the next inner-loop iteration.
    """
    prepare_calls: list[list[str]] = []

    async def prepare_next_turn(turn):
        prepare_calls.append([m.role for m in turn.new_messages])
        return None

    stream_fn = scripted_stream_fn(
        [
            tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "a"})),
            tool_call_response(ToolCall(id="c2", name="echo", arguments={"value": "b"})),
            text_response("done"),
        ]
    )

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool()]),
            make_config(prepare_next_turn=prepare_next_turn),
            None,
            stream_fn,
        )
    )

    # Three turns streamed; the last one ends the run, so preparation only
    # happens before turns two and three.
    assert len(stream_fn.calls) == 3
    assert prepare_calls == [
        ["user", "assistant", "toolResult"],
        ["user", "assistant", "toolResult", "assistant", "toolResult"],
    ]
    assert messages[-1].content[0].text == "done"


async def test_prepare_next_turn_not_called_when_should_stop_after_turn_ends_the_run():
    prepare_calls = {"count": 0}

    async def prepare_next_turn(turn):
        prepare_calls["count"] += 1
        return None

    async def should_stop_after_turn(turn):
        return True

    stream_fn = scripted_stream_fn(
        [
            tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})),
            text_response("should not run"),
        ]
    )

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool()]),
            make_config(prepare_next_turn=prepare_next_turn, should_stop_after_turn=should_stop_after_turn),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 1
    assert prepare_calls["count"] == 0
    assert [m.role for m in messages] == ["user", "assistant", "toolResult"]


async def test_prepare_next_turn_not_called_after_a_natural_final_turn():
    """A turn that ends the run without tool calls is never prepared from."""
    prepare_calls = {"count": 0}

    async def prepare_next_turn(turn):
        prepare_calls["count"] += 1
        return None

    stream_fn = scripted_stream_fn([text_response("done")])

    await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(),
            make_config(prepare_next_turn=prepare_next_turn),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 1
    assert prepare_calls["count"] == 0


async def test_parallel_tool_execution_aborts_before_execute_when_signal_trips_after_preflight():
    """Abort landing between the batch's `before_tool_call` gates and tool start.

    Parallel runners only launch after every preparation in the batch resolves,
    so an abort during a later preparation must stop the earlier, already-prepared
    tools from executing. agent-loop.ts checks `signal?.aborted` at the top of
    each parallel closure and returns an "Operation aborted" result instead of
    running the tool; the port missed that check.
    """
    controller = AbortController()
    executed: list[str] = []
    before_calls: list[str] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(tool_call_id)
        return AgentToolResult(content=[TextContent(text="should not run")], details={})

    async def before_tool_call(context, signal=None):
        before_calls.append(context.tool_call.id)
        if context.tool_call.id == "c2":
            controller.abort()
        return None

    message = make_assistant_message(
        [
            ToolCall(id="c1", name="echo", arguments={"value": "a"}),
            ToolCall(id="c2", name="echo", arguments={"value": "b"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[echo_tool(execute=execute)]),
            make_config(before_tool_call=before_tool_call),
            controller.signal,
            stream_fn,
        )
    )

    assert before_calls == ["c1", "c2"]
    assert executed == []
    tool_ends = [event for event in events if event.type == "tool_execution_end"]
    assert len(tool_ends) == 2
    assert all(event.is_error for event in tool_ends)
    tool_results = [m for m in messages if m.role == "toolResult"]
    assert [m.content[0].text for m in tool_results] == ["Operation aborted", "Operation aborted"]


async def test_transform_context_runs_before_conversion():
    seen: list[int] = []
    transformed: list = []
    converted: list = []

    async def transform_context(messages, signal=None):
        seen.append(len(messages))
        # Keep only the last two messages, like the TypeScript case.
        transformed[:] = messages[-2:]
        return list(transformed)

    def convert(messages):
        converted[:] = default_convert_to_llm(messages)
        return list(converted)

    context = AgentContext(
        system_prompt="You are helpful.",
        messages=[
            UserMessage(content="old message 1"),
            text_response("old response 1"),
            UserMessage(content="old message 2"),
            text_response("old response 2"),
        ],
    )
    stream_fn = scripted_stream_fn([text_response("Response")])
    await collect_events(
        agent_loop(
            [UserMessage(content="new message")],
            context,
            make_config(transform_context=transform_context, convert_to_llm=convert),
            None,
            stream_fn,
        )
    )

    assert seen == [5]
    assert len(transformed) == 2
    # convertToLlm must receive the pruned messages, not the original five.
    assert len(converted) == 2


async def test_get_api_key_is_resolved_per_call():
    async def get_api_key(provider):
        return f"key-for-{provider}"

    stream_fn = scripted_stream_fn([text_response("ok")])
    await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(), make_config(get_api_key=get_api_key), None, stream_fn)
    )

    assert stream_fn.calls[0]["options"].api_key == "key-for-test"


async def test_parallel_tool_calls_run_concurrently():
    order: list[str] = []
    # Event-gated rather than sleep-ordered: c1 cannot finish until c2 has, which is only
    # possible if the two run concurrently. A `sleep(0.02)` would assert the same
    # interleaving by timing luck and can invert under parallel test load; this version
    # deadlocks (and trips the wait_for timeout) instead of silently passing if the loop
    # ever stops running the batch concurrently.
    c2_finished = asyncio.Event()

    async def slow(tool_call_id, params, signal=None, on_update=None):
        order.append(f"start:{tool_call_id}")
        if tool_call_id == "c1":
            await c2_finished.wait()
        order.append(f"end:{tool_call_id}")
        if tool_call_id == "c2":
            c2_finished.set()
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = echo_tool(execute=slow)
    message = make_assistant_message(
        [
            ToolCall(id="c1", name="echo", arguments={"value": "a"}),
            ToolCall(id="c2", name="echo", arguments={"value": "b"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    _events, messages = await asyncio.wait_for(
        collect_events(
            agent_loop(
                [UserMessage(content="hi")],
                AgentContext(tools=[tool]),
                make_config(tool_execution="parallel"),
                None,
                stream_fn,
            )
        ),
        timeout=5.0,
    )

    assert order == ["start:c1", "start:c2", "end:c2", "end:c1"]
    tool_results = [m for m in messages if m.role == "toolResult"]
    # Results keep assistant source order even though c2 finished first.
    assert [m.tool_call_id for m in tool_results] == ["c1", "c2"]


async def test_sequential_mode_runs_tools_one_at_a_time():
    order: list[str] = []

    async def slow(tool_call_id, params, signal=None, on_update=None):
        order.append(f"start:{tool_call_id}")
        await asyncio.sleep(0.02 if tool_call_id == "c1" else 0)
        order.append(f"end:{tool_call_id}")
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = echo_tool(execute=slow)
    message = make_assistant_message(
        [
            ToolCall(id="c1", name="echo", arguments={"value": "a"}),
            ToolCall(id="c2", name="echo", arguments={"value": "b"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[tool]),
            make_config(tool_execution="sequential"),
            None,
            stream_fn,
        )
    )

    assert order == ["start:c1", "end:c1", "start:c2", "end:c2"]


async def test_per_tool_sequential_mode_forces_sequential_batch():
    order: list[str] = []

    async def slow(tool_call_id, params, signal=None, on_update=None):
        order.append(f"start:{tool_call_id}")
        await asyncio.sleep(0.02 if tool_call_id == "c1" else 0)
        order.append(f"end:{tool_call_id}")
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = echo_tool(execute=slow, execution_mode="sequential")
    message = make_assistant_message(
        [
            ToolCall(id="c1", name="echo", arguments={"value": "a"}),
            ToolCall(id="c2", name="echo", arguments={"value": "b"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    events, _messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    assert order == ["start:c1", "end:c1", "start:c2", "end:c2"]
    tool_result_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == "message_end" and event.message.role == "toolResult"
    ]
    assert tool_result_ids == ["c1", "c2"]


async def test_tool_update_callback_emits_events():
    async def with_updates(tool_call_id, params, signal=None, on_update=None):
        if on_update:
            on_update(AgentToolResult(content=[TextContent(text="partial")], details={}))
        return AgentToolResult(content=[TextContent(text="final")], details={})

    tool = echo_tool(execute=with_updates)
    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )

    events, _messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    updates = [event for event in events if event.type == "tool_execution_update"]
    assert len(updates) == 1
    assert updates[0].partial_result.content[0].text == "partial"


async def test_added_tool_names_are_propagated_to_the_result_message():
    async def loader(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="loaded")], details={}, added_tool_names=["extra"])

    tool = echo_tool(execute=loader)
    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )

    _events, messages = await collect_events(
        agent_loop([UserMessage(content="hi")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    tool_result = next(m for m in messages if m.role == "toolResult")
    assert tool_result.added_tool_names == ["extra"]


async def test_agent_loop_continue_requires_messages():
    with pytest.raises(ValueError, match="no messages in context"):
        agent_loop_continue(AgentContext(), make_config(), None, scripted_stream_fn([]))


async def test_agent_loop_continue_rejects_assistant_tail():
    context = AgentContext(messages=[text_response("done")])
    with pytest.raises(ValueError, match="Cannot continue from message role: assistant"):
        agent_loop_continue(context, make_config(), None, scripted_stream_fn([]))


async def test_agent_loop_continue_resumes_from_user_message():
    context = AgentContext(messages=[UserMessage(content="hi")])
    stream_fn = scripted_stream_fn([text_response("resumed")])

    events, messages = await collect_events(agent_loop_continue(context, make_config(), None, stream_fn))

    # Continuation runs do not re-report pre-existing context messages.
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content[0].text == "resumed"
    # No user message events: the key difference from agent_loop.
    message_end_events = [event for event in events if event.type == "message_end"]
    assert len(message_end_events) == 1
    assert message_end_events[0].message.role == "assistant"


async def test_run_agent_loop_emits_to_a_plain_sink():
    collected: list[str] = []

    async def emit(event):
        collected.append(event.type)

    stream_fn = scripted_stream_fn([text_response("ok")])
    messages = await run_agent_loop([UserMessage(content="hi")], AgentContext(), make_config(), emit, None, stream_fn)

    assert collected[0] == "agent_start"
    assert collected[-1] == "agent_end"
    assert len(messages) == 2


# --------------------------------------------------------------------------
# regressions found by review against the TypeScript source
# --------------------------------------------------------------------------


async def test_no_tool_runs_before_every_preflight_gate_has_resolved():
    """Preparation is sequential and completes for the whole batch before any
    tool executes, so a permission gate can still block a later tool."""
    trace: list[str] = []

    async def before_tool_call(context, signal=None):
        trace.append(f"gate-start:{context.tool_call.id}")
        await asyncio.sleep(0.01)
        trace.append(f"gate-end:{context.tool_call.id}")
        return None

    async def execute(tool_call_id, params, signal=None, on_update=None):
        trace.append(f"exec:{tool_call_id}")
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    tool = echo_tool(execute=execute)
    message = make_assistant_message(
        [
            ToolCall(id="c1", name="echo", arguments={"value": "a"}),
            ToolCall(id="c2", name="echo", arguments={"value": "b"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    await collect_events(
        agent_loop(
            [UserMessage(content="hi")],
            AgentContext(tools=[tool]),
            make_config(tool_execution="parallel", before_tool_call=before_tool_call),
            None,
            stream_fn,
        )
    )

    assert trace.index("gate-end:c2") < trace.index("exec:c1")
    assert trace.index("gate-end:c2") < trace.index("exec:c2")


async def test_tool_updates_are_emitted_while_the_tool_is_still_running():
    trace: list[str] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        on_update(AgentToolResult(content=[TextContent(text="partial")], details={}))
        # Yield so an eagerly started sink coroutine can run before we finish.
        await asyncio.sleep(0.01)
        trace.append("tool-done")
        return AgentToolResult(content=[TextContent(text="final")], details={})

    async def emit(event):
        if event.type == "tool_execution_update":
            trace.append("update-emitted")

    tool = echo_tool(execute=execute)
    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={"value": "x"})), text_response("done")]
    )

    await run_agent_loop([UserMessage(content="hi")], AgentContext(tools=[tool]), make_config(), emit, None, stream_fn)

    assert trace == ["update-emitted", "tool-done"]


async def test_parallel_results_still_follow_assistant_source_order():
    first_resolved = False
    parallel_observed = False
    release_first = asyncio.Event()

    async def execute(tool_call_id, params, signal=None, on_update=None):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await release_first.wait()
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={})

    tool = echo_tool(execute=execute)
    message = make_assistant_message(
        [
            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    async def release_later() -> None:
        await asyncio.sleep(0.02)
        release_first.set()

    releaser = asyncio.ensure_future(release_later())
    events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="echo both")],
            AgentContext(tools=[tool]),
            make_config(tool_execution="parallel"),
            None,
            stream_fn,
        )
    )
    await releaser

    tool_execution_end_ids = [event.tool_call_id for event in events if event.type == "tool_execution_end"]
    tool_result_ids = [
        event.message.tool_call_id
        for event in events
        if event.type == "message_end" and event.message.role == "toolResult"
    ]
    turn_tool_result_ids = [
        result.tool_call_id for event in events if event.type == "turn_end" for result in event.tool_results
    ]

    assert parallel_observed is True
    # tool_execution_end fires in completion order...
    assert tool_execution_end_ids == ["tool-2", "tool-1"]
    # ...but results are persisted and reported in assistant source order.
    assert tool_result_ids == ["tool-1", "tool-2"]
    assert turn_tool_result_ids == ["tool-1", "tool-2"]
    tool_results = [m for m in messages if m.role == "toolResult"]
    assert [m.tool_call_id for m in tool_results] == ["tool-1", "tool-2"]


async def test_uses_the_configured_default_when_a_caller_omits_stream_fn():
    calls = 0
    scripted = scripted_stream_fn([text_response("fallback")])

    def counting_stream_fn(model, context, options=None):
        nonlocal calls
        calls += 1
        return scripted(model, context, options)

    set_default_stream_fn(counting_stream_fn)
    try:
        stream = agent_loop([UserMessage(content="Hello")], AgentContext(), make_config())
        await stream.result()
        assert calls == 1
    finally:
        set_default_stream_fn(None)


async def test_custom_message_types_are_filtered_by_convert_to_llm():
    notification = _RoleMessage(role="notification", content="This is a notification")
    converted: list = []

    def convert(messages):
        nonlocal converted
        converted = default_convert_to_llm([m for m in messages if m.role != "notification"])
        return converted

    stream_fn = scripted_stream_fn([text_response("Response")])
    context = AgentContext(system_prompt="You are helpful.", messages=[notification])

    await collect_events(
        agent_loop([UserMessage(content="Hello")], context, make_config(convert_to_llm=convert), None, stream_fn)
    )

    assert len(converted) == 1
    assert converted[0].role == "user"


async def test_mutated_before_tool_call_args_execute_without_revalidation():
    executed: list = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={})

    async def before_tool_call(context, signal=None):
        # Replacing a validated string with an int would fail schema validation if it re-ran.
        context.args["value"] = 123
        return None

    tool = echo_tool(execute=execute)
    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="tool-1", name="echo", arguments={"value": "hello"})), text_response("done")]
    )

    await collect_events(
        agent_loop(
            [UserMessage(content="echo something")],
            AgentContext(tools=[tool]),
            make_config(before_tool_call=before_tool_call),
            None,
            stream_fn,
        )
    )

    assert executed == [123]


async def test_prepare_arguments_runs_before_validation():
    executed: list = []

    def prepare_arguments(args):
        if not isinstance(args, dict):
            return args
        if not isinstance(args.get("oldText"), str) or not isinstance(args.get("newText"), str):
            return args
        return {"edits": [*args.get("edits", []), {"oldText": args["oldText"], "newText": args["newText"]}]}

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(params["edits"])
        return AgentToolResult(content=[TextContent(text=f"edited {len(params['edits'])}")], details={})

    tool = AgentTool(
        name="edit",
        label="Edit",
        description="Edit tool",
        parameters={
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
                        "required": ["oldText", "newText"],
                    },
                }
            },
            "required": ["edits"],
        },
        prepare_arguments=prepare_arguments,
        execute=execute,
    )
    stream_fn = scripted_stream_fn(
        [
            tool_call_response(ToolCall(id="tool-1", name="edit", arguments={"oldText": "before", "newText": "after"})),
            text_response("done"),
        ]
    )

    await collect_events(
        agent_loop([UserMessage(content="edit something")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    assert executed == [[{"oldText": "before", "newText": "after"}]]


async def test_one_sequential_tool_in_a_batch_forces_the_whole_batch_sequential():
    execution_order: list[str] = []
    release_slow = asyncio.Event()

    async def slow_execute(tool_call_id, params, signal=None, on_update=None):
        execution_order.append(f"slow:{params['value']}")
        if params["value"] == "a":
            await release_slow.wait()
        return AgentToolResult(content=[TextContent(text=f"slow: {params['value']}")], details={})

    async def fast_execute(tool_call_id, params, signal=None, on_update=None):
        execution_order.append(f"fast:{params['value']}")
        return AgentToolResult(content=[TextContent(text=f"fast: {params['value']}")], details={})

    slow_tool = echo_tool("slow", execute=slow_execute, execution_mode="sequential")
    fast_tool = echo_tool("fast", execute=fast_execute)
    message = make_assistant_message(
        [
            ToolCall(id="tool-1", name="slow", arguments={"value": "a"}),
            ToolCall(id="tool-2", name="fast", arguments={"value": "b"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    async def release_later() -> None:
        await asyncio.sleep(0.02)
        release_slow.set()

    releaser = asyncio.ensure_future(release_later())
    await collect_events(
        agent_loop(
            [UserMessage(content="run both")],
            AgentContext(tools=[slow_tool, fast_tool]),
            make_config(),
            None,
            stream_fn,
        )
    )
    await releaser

    # Fast tool must not run before the slow tool finishes.
    assert execution_order[0] == "slow:a"
    assert "fast:b" in execution_order


async def test_explicit_parallel_execution_mode_allows_overlap():
    first_resolved = False
    parallel_observed = False
    release_first = asyncio.Event()

    async def execute(tool_call_id, params, signal=None, on_update=None):
        nonlocal first_resolved, parallel_observed
        if params["value"] == "first":
            await release_first.wait()
            first_resolved = True
        if params["value"] == "second" and not first_resolved:
            parallel_observed = True
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={})

    tool = echo_tool(execute=execute, execution_mode="parallel")
    message = make_assistant_message(
        [
            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    async def release_later() -> None:
        await asyncio.sleep(0.02)
        release_first.set()

    releaser = asyncio.ensure_future(release_later())
    await collect_events(
        agent_loop([UserMessage(content="echo both")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )
    await releaser

    assert parallel_observed is True


async def test_stops_after_a_tool_batch_when_every_result_terminates():
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={}, terminate=True)

    tool = echo_tool(execute=execute)
    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="tool-1", name="echo", arguments={"value": "hello"}))]
    )

    events, messages = await collect_events(
        agent_loop([UserMessage(content="echo something")], AgentContext(tools=[tool]), make_config(), None, stream_fn)
    )

    assert len(stream_fn.calls) == 1
    assert [message.role for message in messages] == ["user", "assistant", "toolResult"]
    assert len([event for event in events if event.type == "turn_end"]) == 1


async def test_continues_after_a_mixed_batch_with_one_terminating_blocked_call():
    executed: list[str] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        executed.append(params["value"])
        return AgentToolResult(content=[TextContent(text=f"echoed: {params['value']}")], details={})

    async def before_tool_call(context, signal=None):
        if context.args["value"] == "first":
            return BeforeToolCallResult(block=True, reason="Blocked first", terminate=True)
        return None

    tool = echo_tool(execute=execute)
    message = make_assistant_message(
        [
            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    await collect_events(
        agent_loop(
            [UserMessage(content="echo both")],
            AgentContext(tools=[tool]),
            make_config(tool_execution="parallel", before_tool_call=before_tool_call),
            None,
            stream_fn,
        )
    )

    assert executed == ["second"]
    assert len(stream_fn.calls) == 2


async def test_continues_after_parallel_tool_calls_when_not_all_results_terminate():
    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(text=f"echoed: {params['value']}")],
            details={},
            terminate=params["value"] == "first",
        )

    tool = echo_tool(execute=execute)
    message = make_assistant_message(
        [
            ToolCall(id="tool-1", name="echo", arguments={"value": "first"}),
            ToolCall(id="tool-2", name="echo", arguments={"value": "second"}),
        ],
        stop_reason="toolUse",
    )
    stream_fn = scripted_stream_fn([message, text_response("done")])

    _events, messages = await collect_events(
        agent_loop(
            [UserMessage(content="echo both")],
            AgentContext(tools=[tool]),
            make_config(tool_execution="parallel"),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 2
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "toolResult",
        "toolResult",
        "assistant",
    ]


async def test_after_tool_call_can_mark_a_tool_batch_as_terminating():
    async def after_tool_call(context, signal=None):
        return AfterToolCallResult(terminate=True)

    stream_fn = scripted_stream_fn(
        [tool_call_response(ToolCall(id="tool-1", name="echo", arguments={"value": "hello"}))]
    )

    await collect_events(
        agent_loop(
            [UserMessage(content="echo something")],
            AgentContext(tools=[echo_tool()]),
            make_config(after_tool_call=after_tool_call),
            None,
            stream_fn,
        )
    )

    assert len(stream_fn.calls) == 1


async def test_agent_loop_continue_accepts_a_custom_message_tail():
    custom = _RoleMessage(role="custom", content="Hook content")

    def convert(messages):
        return default_convert_to_llm([UserMessage(content=m.content) if m.role == "custom" else m for m in messages])

    stream_fn = scripted_stream_fn([text_response("Response to custom message")])
    context = AgentContext(system_prompt="You are helpful.", messages=[custom])

    _events, messages = await collect_events(
        agent_loop_continue(context, make_config(convert_to_llm=convert), None, stream_fn)
    )

    assert len(messages) == 1
    assert messages[0].role == "assistant"
