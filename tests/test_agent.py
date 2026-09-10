"""Tests for `pi_agent.agent`.

Ported from `packages/agent/test/agent.test.ts`, extended to cover the facade
paths that the TypeScript suite only exercises indirectly (queue drain modes,
reset guards, continuation from every tail role).

Every TypeScript `it(...)` has a counterpart here, including the
"unhandledRejection" bookkeeping in "should ignore tool updates after the tool
execution settles". Python has no `process.on("unhandledRejection")`, but the
loop dispatches `on_update` through `spawn(...)`, so the equivalent failure is an
orphaned task whose exception is never retrieved (plus "coroutine was never
awaited" warnings). `capture_unhandled_async_errors` below records both, which is
what `expect(unhandledRejections).toEqual([])` is actually asserting.

Every await that could otherwise block forever is wrapped in
`asyncio.wait_for`; no test performs network or real-home I/O.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import warnings
from dataclasses import replace

import pytest
from agent_helpers import (
    TEST_MODEL,
    make_assistant_message,
    scripted_stream_fn,
    text_response,
    tool_call_response,
)
from pi_ai import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    StartEvent,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_ai.utils.tasks import spawn

from pi_agent.agent import DEFAULT_MODEL, Agent, MutableAgentState, default_convert_to_llm
from pi_agent.harness.messages import CustomMessage
from pi_agent.stream_fn import set_default_stream_fn
from pi_agent.types import AgentTool, AgentToolResult

TIMEOUT = 5


def unused_stream_fn(model, context, options=None):
    raise AssertionError("Unexpected stream call")


def make_agent(responses: list[AssistantMessage] | None = None, **kwargs) -> Agent:
    stream_fn = scripted_stream_fn(responses) if responses is not None else unused_stream_fn
    kwargs.setdefault("initial_state", MutableAgentState(model=TEST_MODEL))
    return Agent(stream_fn, **kwargs)


def gated_stream_fn(started: asyncio.Event, release: asyncio.Event, message: AssistantMessage):
    """A stream that opens, then waits for ``release`` before completing."""

    def stream_fn(model, context, options=None):
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=make_assistant_message([TextContent(text="")])))

        async def finish() -> None:
            started.set()
            await release.wait()
            stream.push(DoneEvent(reason=message.stop_reason, message=message))
            stream.end()

        spawn(finish())
        return stream

    return stream_fn


def abortable_stream_fn(started: asyncio.Event):
    """A stream that only completes once the run's abort signal fires."""

    def stream_fn(model, context, options=None):
        signal = options.signal
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=make_assistant_message([TextContent(text="")])))

        async def finish() -> None:
            started.set()
            await signal.wait()
            aborted = make_assistant_message([], stop_reason="aborted", error_message="Aborted")
            stream.push(ErrorEvent(reason="aborted", error=aborted))
            stream.end()

        spawn(finish())
        return stream

    return stream_fn


def echo_tool(execute=None) -> AgentTool:
    async def default_execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    return AgentTool(
        name="echo",
        description="Echo a value",
        parameters={"type": "object", "properties": {}},
        label="echo",
        execute=execute or default_execute,
    )


def user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=now_ms())


def texts_of(message) -> list[str]:
    if isinstance(message.content, str):
        return [message.content]
    return [block.text for block in message.content if block.type == "text"]


# ---------------------------------------------------------------------------
# construction and state
# ---------------------------------------------------------------------------


async def test_default_state():
    agent = Agent(unused_stream_fn)

    assert agent.state.system_prompt == ""
    assert agent.state.model is DEFAULT_MODEL
    assert agent.state.thinking_level == "off"
    assert agent.state.tools == []
    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == set()
    assert agent.state.error_message is None
    assert agent.convert_to_llm is default_convert_to_llm
    assert agent.steering_mode == "one-at-a-time"
    assert agent.follow_up_mode == "one-at-a-time"
    assert agent.signal is None


async def test_custom_initial_state_is_used_as_is():
    state = MutableAgentState(system_prompt="You are a helpful assistant.", model=TEST_MODEL, thinking_level="low")
    agent = Agent(unused_stream_fn, initial_state=state)

    assert agent.state is state
    assert agent.state.system_prompt == "You are a helpful assistant."
    assert agent.state.model is TEST_MODEL
    assert agent.state.thinking_level == "low"


async def test_state_setters_copy_the_assigned_lists():
    agent = make_agent()

    agent.state.system_prompt = "Custom prompt"
    assert agent.state.system_prompt == "Custom prompt"

    new_model = replace(DEFAULT_MODEL, id="gemini-2.5-flash")
    agent.state.model = new_model
    assert agent.state.model is new_model

    agent.state.thinking_level = "high"
    assert agent.state.thinking_level == "high"

    tools = [echo_tool()]
    agent.state.tools = tools
    assert agent.state.tools == tools
    assert agent.state.tools is not tools

    messages = [user_message("Hello")]
    agent.state.messages = messages
    assert agent.state.messages == messages
    assert agent.state.messages is not messages

    agent.state.messages.append(text_response("Hi"))
    assert len(agent.state.messages) == 2

    agent.state.messages = []
    assert agent.state.messages == []


# ---------------------------------------------------------------------------
# subscription
# ---------------------------------------------------------------------------


async def test_subscribe_delivers_events_and_unsubscribe_stops_them():
    agent = make_agent([text_response("one"), text_response("two")])
    seen: list[str] = []
    unsubscribe = agent.subscribe(lambda event, signal: seen.append(event.type))

    # Subscribing alone emits nothing, and state mutation is not an event.
    assert seen == []
    agent.state.system_prompt = "Test prompt"
    assert seen == []

    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)
    assert seen == [
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

    unsubscribe()
    count_after_first_run = len(seen)
    await asyncio.wait_for(agent.prompt("again"), timeout=TIMEOUT)
    assert len(seen) == count_after_first_run

    # Unsubscribing twice is a no-op rather than an error.
    unsubscribe()


async def test_listeners_receive_the_active_run_signal():
    started = asyncio.Event()
    agent = make_agent()
    agent.stream_function = abortable_stream_fn(started)
    received: list = []
    agent.subscribe(lambda event, signal: received.append(signal) if event.type == "agent_start" else None)

    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)

    assert received and received[0] is agent.signal
    assert received[0].aborted is False

    agent.abort()
    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)
    assert received[0].aborted is True


async def test_prompt_waits_for_async_listeners_to_settle():
    release = asyncio.Event()
    agent = make_agent([text_response("ok")])

    finished: list[str] = []

    async def listener(event, signal):
        if event.type == "agent_end":
            await release.wait()
            finished.append("listener")

    agent.subscribe(listener)
    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.sleep(0.05)

    assert prompt_task.done() is False
    assert finished == []
    assert agent.state.is_streaming is True

    release.set()
    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)

    assert finished == ["listener"]
    assert agent.state.is_streaming is False


async def test_process_events_outside_a_run_is_rejected():
    agent = make_agent()
    from pi_agent.types import AgentStartEvent

    with pytest.raises(RuntimeError, match="Agent listener invoked outside active run"):
        await asyncio.wait_for(agent._process_events(AgentStartEvent()), timeout=TIMEOUT)


# ---------------------------------------------------------------------------
# prompt input handling
# ---------------------------------------------------------------------------


async def test_prompt_from_text_builds_a_user_message():
    agent = make_agent([text_response("ok")])
    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    first = agent.state.messages[0]
    assert first.role == "user"
    assert texts_of(first) == ["hello"]
    assert first.timestamp > 0


async def test_prompt_from_text_appends_images():
    agent = make_agent([text_response("ok")])
    image = ImageContent(data="Zm9v", mime_type="image/png")

    await asyncio.wait_for(agent.prompt("look", [image]), timeout=TIMEOUT)

    first = agent.state.messages[0]
    assert [block.type for block in first.content] == ["text", "image"]
    assert first.content[1] is image


async def test_prompt_accepts_a_single_message_and_a_batch():
    agent = make_agent([text_response("one"), text_response("two")])

    single = user_message("single")
    await asyncio.wait_for(agent.prompt(single), timeout=TIMEOUT)
    assert agent.state.messages[0] is single

    batch = [user_message("a"), user_message("b")]
    await asyncio.wait_for(agent.prompt(batch), timeout=TIMEOUT)
    assert [texts_of(m) for m in agent.state.messages[2:4]] == [["a"], ["b"]]
    assert [m.role for m in agent.state.messages] == ["user", "assistant", "user", "user", "assistant"]


# ---------------------------------------------------------------------------
# in-flight guards
# ---------------------------------------------------------------------------


async def test_prompt_while_processing_raises():
    started = asyncio.Event()
    release = asyncio.Event()
    agent = make_agent()
    agent.stream_function = gated_stream_fn(started, release, text_response("Done"))

    first = asyncio.ensure_future(agent.prompt("First message"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)
    assert agent.state.is_streaming is True

    with pytest.raises(RuntimeError, match="Agent is already processing a prompt"):
        await asyncio.wait_for(agent.prompt("Second message"), timeout=TIMEOUT)

    release.set()
    await asyncio.wait_for(first, timeout=TIMEOUT)


async def test_continue_while_processing_raises():
    started = asyncio.Event()
    release = asyncio.Event()
    agent = make_agent()
    agent.stream_function = gated_stream_fn(started, release, text_response("Done"))

    first = asyncio.ensure_future(agent.prompt("First message"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)

    with pytest.raises(
        RuntimeError, match=re.escape("Agent is already processing. Wait for completion before continuing.")
    ):
        await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    release.set()
    await asyncio.wait_for(first, timeout=TIMEOUT)


async def test_reset_while_processing_raises_and_keeps_the_transcript():
    started = asyncio.Event()
    release = asyncio.Event()
    agent = make_agent()
    agent.stream_function = gated_stream_fn(started, release, text_response("Done"))

    prompt_task = asyncio.ensure_future(agent.prompt("Hello"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)

    try:
        assert agent.state.is_streaming is True
        assert [m.role for m in agent.state.messages] == ["user"]
        with pytest.raises(RuntimeError, match="Wait for completion before resetting"):
            agent.reset()
        assert agent.state.is_streaming is True
        assert [m.role for m in agent.state.messages] == ["user"]
    finally:
        release.set()
        await asyncio.wait_for(prompt_task, timeout=TIMEOUT)

    assert agent.state.is_streaming is False
    assert [m.role for m in agent.state.messages] == ["user", "assistant"]


async def test_reset_clears_transcript_runtime_state_and_queues():
    agent = make_agent([text_response("ok")])
    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    agent.state.error_message = "boom"
    agent.state.streaming_message = text_response("partial")
    agent.state.pending_tool_calls = {"c1"}
    agent.state.is_streaming = True
    agent.steer(user_message("steer"))
    agent.follow_up(user_message("follow"))

    agent.reset()

    assert agent.state.messages == []
    assert agent.state.is_streaming is False
    assert agent.state.streaming_message is None
    assert agent.state.pending_tool_calls == set()
    assert agent.state.error_message is None
    assert agent.has_queued_messages() is False


# ---------------------------------------------------------------------------
# abort and idle
# ---------------------------------------------------------------------------


async def test_abort_while_idle_is_a_noop():
    agent = make_agent()
    agent.abort()
    assert agent.signal is None


async def test_abort_mid_turn_ends_the_run_with_an_aborted_message():
    started = asyncio.Event()
    agent = make_agent()
    agent.stream_function = abortable_stream_fn(started)
    seen: list[str] = []
    agent.subscribe(lambda event, signal: seen.append(event.type))

    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)
    assert agent.state.is_streaming is True

    agent.abort()
    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)

    assert agent.state.is_streaming is False
    assert agent.signal is None
    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert last.stop_reason == "aborted"
    assert agent.state.error_message == "Aborted"
    assert seen[-2:] == ["turn_end", "agent_end"]


async def test_wait_for_idle_returns_immediately_when_idle():
    agent = make_agent()
    await asyncio.wait_for(agent.wait_for_idle(), timeout=TIMEOUT)


async def test_wait_for_idle_waits_for_the_active_run():
    release = asyncio.Event()
    agent = make_agent([text_response("ok")])

    async def listener(event, signal):
        if event.type == "message_end" and event.message.role == "assistant":
            await release.wait()

    agent.subscribe(listener)

    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.sleep(0)
    idle_task = asyncio.ensure_future(agent.wait_for_idle())
    await asyncio.sleep(0.05)

    assert idle_task.done() is False
    assert agent.state.is_streaming is True

    release.set()
    await asyncio.wait_for(asyncio.gather(prompt_task, idle_task), timeout=TIMEOUT)

    assert agent.state.is_streaming is False


# ---------------------------------------------------------------------------
# run failures
# ---------------------------------------------------------------------------


async def test_thrown_stream_failure_emits_the_full_lifecycle():
    def exploding_stream_fn(model, context, options=None):
        raise RuntimeError("provider exploded")

    agent = Agent(exploding_stream_fn, initial_state=MutableAgentState(model=TEST_MODEL))
    events: list[str] = []
    agent.subscribe(lambda event, signal: events.append(event.type))

    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    assert events == [
        "agent_start",
        "turn_start",
        "message_start",
        "message_end",
        "message_start",
        "message_end",
        "turn_end",
        "agent_end",
    ]
    last = agent.state.messages[-1]
    assert last.role == "assistant"
    assert last.stop_reason == "error"
    assert last.error_message == "provider exploded"
    assert last.api == TEST_MODEL.api
    assert last.provider == TEST_MODEL.provider
    assert last.model == TEST_MODEL.id
    assert agent.state.error_message == "provider exploded"
    assert agent.state.is_streaming is False


async def test_failure_after_abort_is_reported_as_aborted():
    started = asyncio.Event()

    async def failing_stream_fn(model, context, options=None):
        started.set()
        await options.signal.wait()
        raise RuntimeError("connection dropped")

    agent = Agent(failing_stream_fn, initial_state=MutableAgentState(model=TEST_MODEL))
    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)

    agent.abort()
    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)

    last = agent.state.messages[-1]
    assert last.stop_reason == "aborted"
    assert last.error_message == "connection dropped"


async def test_error_stop_reason_records_the_error_message():
    agent = make_agent([make_assistant_message([], stop_reason="error", error_message="boom")])

    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    assert agent.state.error_message == "boom"
    assert agent.state.messages[-1].stop_reason == "error"


# ---------------------------------------------------------------------------
# continuation
# ---------------------------------------------------------------------------


async def test_continue_without_messages_raises():
    agent = make_agent()
    with pytest.raises(RuntimeError, match="No messages to continue from"):
        await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)


async def test_continue_from_a_user_tail_resumes_the_run():
    agent = make_agent([text_response("resumed")])
    agent.state.messages = [user_message("hi")]

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert [m.role for m in agent.state.messages] == ["user", "assistant"]
    assert texts_of(agent.state.messages[-1]) == ["resumed"]


async def test_continue_from_a_tool_result_tail_resumes_the_run():
    agent = make_agent([text_response("after tool")])
    agent.state.messages = [
        user_message("hi"),
        tool_call_response(ToolCall(id="c1", name="echo", arguments={})),
        ToolResultMessage(tool_call_id="c1", tool_name="echo", content=[TextContent(text="done")]),
    ]

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert [m.role for m in agent.state.messages] == ["user", "assistant", "toolResult", "assistant"]
    assert texts_of(agent.state.messages[-1]) == ["after tool"]


async def test_continue_from_a_custom_role_tail_resumes_the_run():
    agent = make_agent([text_response("after custom")])
    agent.state.messages = [
        user_message("hi"),
        CustomMessage(custom_type="note", content="a note", display=True, timestamp=now_ms()),
    ]

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert [m.role for m in agent.state.messages] == ["user", "custom", "assistant"]


async def test_continue_from_an_assistant_tail_without_queues_raises():
    agent = make_agent()
    agent.state.messages = [user_message("hi"), text_response("done")]

    with pytest.raises(RuntimeError, match="Cannot continue from message role: assistant"):
        await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)


async def test_continue_from_an_assistant_tail_drains_steering_one_at_a_time():
    agent = make_agent([text_response("Processed 1"), text_response("Processed 2")])
    agent.state.messages = [user_message("Initial"), text_response("Initial response")]

    agent.steer(user_message("Steering 1"))
    agent.steer(user_message("Steering 2"))

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert [m.role for m in agent.state.messages[-4:]] == ["user", "assistant", "user", "assistant"]
    assert [texts_of(m) for m in agent.state.messages[-4:]] == [
        ["Steering 1"],
        ["Processed 1"],
        ["Steering 2"],
        ["Processed 2"],
    ]
    assert agent.has_queued_messages() is False


async def test_continue_from_an_assistant_tail_drains_all_steering_at_once():
    agent = make_agent([text_response("Processed")], steering_mode="all")
    agent.state.messages = [user_message("Initial"), text_response("Initial response")]

    agent.steer(user_message("Steering 1"))
    agent.steer(user_message("Steering 2"))

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert [m.role for m in agent.state.messages[-3:]] == ["user", "user", "assistant"]
    assert [texts_of(m) for m in agent.state.messages[-3:]] == [["Steering 1"], ["Steering 2"], ["Processed"]]


async def test_continue_from_an_assistant_tail_drains_a_follow_up():
    agent = make_agent([text_response("Processed")])
    agent.state.messages = [user_message("Initial"), text_response("Initial response")]

    agent.follow_up(user_message("Queued follow-up"))

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    assert [texts_of(m) for m in agent.state.messages[-2:]] == [["Queued follow-up"], ["Processed"]]
    assert agent.state.messages[-1].role == "assistant"


async def test_steering_is_drained_before_follow_ups():
    agent = make_agent([text_response("Processed steering"), text_response("Processed follow-up")])
    agent.state.messages = [user_message("Initial"), text_response("Initial response")]

    agent.steer(user_message("Steering"))
    agent.follow_up(user_message("Follow-up"))

    await asyncio.wait_for(agent.continue_(), timeout=TIMEOUT)

    # The steering message starts the run; the follow-up only restarts the loop
    # once the agent would otherwise have stopped.
    assert [texts_of(m) for m in agent.state.messages[-4:]] == [
        ["Steering"],
        ["Processed steering"],
        ["Follow-up"],
        ["Processed follow-up"],
    ]
    assert agent.has_queued_messages() is False


# ---------------------------------------------------------------------------
# queues
# ---------------------------------------------------------------------------


async def test_queue_mode_accessors():
    agent = make_agent()

    assert agent.steering_mode == "one-at-a-time"
    agent.steering_mode = "all"
    assert agent.steering_mode == "all"

    assert agent.follow_up_mode == "one-at-a-time"
    agent.follow_up_mode = "all"
    assert agent.follow_up_mode == "all"


async def test_queued_messages_are_not_in_the_transcript_until_drained():
    agent = make_agent()

    steering = user_message("Steering message")
    follow_up = user_message("Follow-up message")
    agent.steer(steering)
    agent.follow_up(follow_up)

    assert agent.state.messages == []
    assert agent.has_queued_messages() is True

    agent.clear_steering_queue()
    assert agent.has_queued_messages() is True
    agent.clear_follow_up_queue()
    assert agent.has_queued_messages() is False

    agent.steer(steering)
    agent.follow_up(follow_up)
    agent.clear_all_queues()
    assert agent.has_queued_messages() is False


async def test_steering_queued_during_a_run_joins_the_next_turn():
    started = asyncio.Event()
    release = asyncio.Event()
    agent = make_agent()
    replies = scripted_stream_fn([text_response("second")])
    first_call = {"done": False}

    def stream_fn(model, context, options=None):
        if first_call["done"]:
            return replies(model, context, options)
        first_call["done"] = True
        return gated_stream_fn(started, release, text_response("first"))(model, context, options)

    agent.stream_function = stream_fn

    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)
    agent.steer(user_message("mid-run steer"))
    release.set()

    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)

    assert [m.role for m in agent.state.messages] == ["user", "assistant", "user", "assistant"]
    assert texts_of(agent.state.messages[2]) == ["mid-run steer"]
    assert texts_of(agent.state.messages[3]) == ["second"]


async def test_follow_up_queued_during_a_run_restarts_the_loop():
    started = asyncio.Event()
    release = asyncio.Event()
    agent = make_agent(follow_up_mode="all")
    replies = scripted_stream_fn([text_response("second")])
    first_call = {"done": False}

    def stream_fn(model, context, options=None):
        if first_call["done"]:
            return replies(model, context, options)
        first_call["done"] = True
        return gated_stream_fn(started, release, text_response("first"))(model, context, options)

    agent.stream_function = stream_fn

    prompt_task = asyncio.ensure_future(agent.prompt("hello"))
    await asyncio.wait_for(started.wait(), timeout=TIMEOUT)
    agent.follow_up(user_message("queued follow-up"))
    release.set()

    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)

    assert [texts_of(m) for m in agent.state.messages[-2:]] == [["queued follow-up"], ["second"]]


# ---------------------------------------------------------------------------
# forwarding into the loop configuration
# ---------------------------------------------------------------------------


async def test_stream_options_carry_the_facade_configuration():
    def on_payload(payload):
        return None

    def on_response(response):
        return None

    agent = make_agent(
        [text_response("ok")],
        session_id="session-abc",
        transport="fetch",
        max_retry_delay_ms=1234,
        tool_execution="sequential",
        on_payload=on_payload,
        on_response=on_response,
        get_api_key=lambda provider: f"key-for-{provider}",
    )
    agent.state.thinking_level = "low"

    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    options = agent.stream_function.calls[0]["options"]
    assert options.session_id == "session-abc"
    assert options.transport == "fetch"
    assert options.max_retry_delay_ms == 1234
    assert options.tool_execution == "sequential"
    assert options.on_payload is on_payload
    assert options.on_response is on_response
    assert options.api_key == "key-for-test"
    assert options.reasoning == "low"
    assert agent.stream_function.calls[0]["model"] is TEST_MODEL


async def test_thinking_level_off_sends_no_reasoning():
    agent = make_agent([text_response("ok")])
    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    assert agent.stream_function.calls[0]["options"].reasoning is None


async def test_session_id_can_be_changed_between_runs():
    agent = make_agent([text_response("one"), text_response("two")], session_id="session-abc")

    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)
    assert agent.stream_function.calls[0]["options"].session_id == "session-abc"

    agent.session_id = "session-def"
    await asyncio.wait_for(agent.prompt("hello again"), timeout=TIMEOUT)
    assert agent.stream_function.calls[1]["options"].session_id == "session-def"


async def test_context_snapshot_uses_the_current_state():
    agent = make_agent([text_response("ok")])
    agent.state.system_prompt = "sys"
    agent.state.tools = [echo_tool()]

    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    context = agent.stream_function.calls[0]["context"]
    assert context.system_prompt == "sys"
    assert [tool.name for tool in context.tools] == ["echo"]


async def test_transform_context_is_forwarded_with_the_run_signal():
    seen: list[int] = []

    async def transform_context(messages, signal=None):
        seen.append(len(messages))
        assert signal is not None
        return messages

    agent = make_agent([text_response("ok")], transform_context=transform_context)
    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    assert seen == [1]


async def test_convert_to_llm_override_is_used():
    calls: list[int] = []

    def convert(messages):
        calls.append(len(messages))
        return default_convert_to_llm(messages)

    agent = make_agent([text_response("ok")], convert_to_llm=convert)
    await asyncio.wait_for(agent.prompt("hello"), timeout=TIMEOUT)

    assert calls == [1]


async def test_should_stop_after_turn_receives_the_active_signal():
    seen_signals: list = []
    callback_context_roles: list[str] = []

    def should_stop(context, signal):
        seen_signals.append(signal)
        callback_context_roles[:] = [message.role for message in context.context.messages]
        return True

    agent = make_agent([tool_call_response(ToolCall(id="c1", name="echo", arguments={}))])
    agent.should_stop_after_turn = should_stop
    agent.state.tools = [echo_tool()]

    await asyncio.wait_for(agent.prompt("start"), timeout=TIMEOUT)

    assert len(agent.stream_function.calls) == 1
    assert seen_signals and seen_signals[0] is not None
    assert callback_context_roles == ["user", "assistant", "toolResult"]
    assert [m.role for m in agent.state.messages] == ["user", "assistant", "toolResult"]


async def test_prepare_next_turn_receives_the_active_signal():
    seen_signals: list = []

    async def prepare_next_turn(signal):
        seen_signals.append(signal)
        return None

    agent = make_agent(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={})), text_response("done")],
        prepare_next_turn=prepare_next_turn,
    )
    agent.state.tools = [echo_tool()]

    await asyncio.wait_for(agent.prompt("start"), timeout=TIMEOUT)

    assert len(agent.stream_function.calls) == 2
    assert seen_signals and seen_signals[0] is not None


async def test_prepare_next_turn_with_context_takes_precedence():
    legacy_calls: list[int] = []
    context_calls: list[list[str]] = []

    def legacy(signal):
        legacy_calls.append(1)
        return None

    def with_context(context, signal):
        context_calls.append([m.role for m in context.new_messages])
        return None

    # Two turns: `prepare_next_turn` only runs at the start of a continuing
    # turn (matching `runLoop` in agent-loop.ts), so a single-turn run would
    # never invoke it.
    agent = make_agent(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={})), text_response("done")],
        prepare_next_turn=legacy,
        prepare_next_turn_with_context=with_context,
    )
    agent.state.tools = [echo_tool()]

    await asyncio.wait_for(agent.prompt("start"), timeout=TIMEOUT)

    assert legacy_calls == []
    assert context_calls == [["user", "assistant", "toolResult"]]


async def test_prepare_next_turn_can_swap_the_model():
    from pi_ai import Model

    other_model = Model(id="other", name="Other", api="test-api", provider="test", base_url="")

    from pi_agent.types import AgentLoopTurnUpdate

    def with_context(context, signal):
        return AgentLoopTurnUpdate(model=other_model)

    agent = make_agent(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={})), text_response("done")],
        prepare_next_turn_with_context=with_context,
    )
    agent.state.tools = [echo_tool()]

    await asyncio.wait_for(agent.prompt("start"), timeout=TIMEOUT)

    assert agent.stream_function.calls[1]["model"] is other_model


async def test_before_and_after_tool_call_hooks_are_forwarded():
    from pi_agent.types import AfterToolCallResult, BeforeToolCallResult

    before_seen: list[str] = []

    async def before_tool_call(context, signal=None):
        before_seen.append(context.tool_call.id)
        return BeforeToolCallResult()

    async def after_tool_call(context, signal=None):
        return AfterToolCallResult(content=[TextContent(text="overridden")])

    agent = make_agent(
        [tool_call_response(ToolCall(id="c1", name="echo", arguments={})), text_response("done")],
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
    )
    agent.state.tools = [echo_tool()]

    await asyncio.wait_for(agent.prompt("start"), timeout=TIMEOUT)

    assert before_seen == ["c1"]
    tool_result = next(m for m in agent.state.messages if m.role == "toolResult")
    assert texts_of(tool_result) == ["overridden"]


async def test_pending_tool_calls_track_in_flight_tools():
    snapshots: list[set[str]] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        snapshots.append(set(agent.state.pending_tool_calls))
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    agent = make_agent([tool_call_response(ToolCall(id="c1", name="echo", arguments={})), text_response("done")])
    agent.state.tools = [echo_tool(execute=execute)]

    await asyncio.wait_for(agent.prompt("run tool"), timeout=TIMEOUT)

    assert snapshots == [{"c1"}]
    assert agent.state.pending_tool_calls == set()
    assert agent.state.streaming_message is None


async def test_uses_the_configured_default_when_a_caller_omits_stream_fn():
    calls = 0
    scripted = scripted_stream_fn([text_response("fallback")])

    def counting_stream_fn(model, context, options=None):
        nonlocal calls
        calls += 1
        return scripted(model, context, options)

    set_default_stream_fn(counting_stream_fn)
    try:
        agent = Agent(initial_state=MutableAgentState(model=TEST_MODEL))
        await asyncio.wait_for(agent.prompt("Hello"), timeout=TIMEOUT)
        assert calls == 1
    finally:
        set_default_stream_fn(None)


@contextlib.asynccontextmanager
async def capture_unhandled_async_errors():
    """Python analogue of `process.on("unhandledRejection", ...)`.

    The loop dispatches `on_update` through `spawn(...)`, so a callback that blows up
    becomes a task whose exception nobody retrieves -- Python's version of an unhandled
    rejection. Relying on garbage collection to surface those is unreliable (a test that
    holds the callback closure keeps the task alive), so every task created inside the
    window is recorded via the loop's task factory and inspected directly. "Coroutine was
    never awaited" warnings are folded in as the second way an async callback goes missing.
    """
    loop = asyncio.get_event_loop()
    previous_factory = loop.get_task_factory()
    previous_handler = loop.get_exception_handler()
    spawned: list[asyncio.Task[object]] = []
    captured: list[object] = []

    def task_factory(active_loop, coro, **kwargs):
        task = asyncio.Task(coro, loop=active_loop, **kwargs)
        spawned.append(task)
        return task

    def handler(_loop, context):
        captured.append(context.get("exception") or context.get("message"))

    loop.set_task_factory(task_factory)
    loop.set_exception_handler(handler)
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        try:
            yield captured
        finally:
            loop.set_task_factory(previous_factory)
            loop.set_exception_handler(previous_handler)
            # Wait for the recorded tasks to actually settle rather than yielding a fixed
            # number of times. `await asyncio.sleep(0)` advances exactly one loop iteration,
            # so `for _ in range(3)` is not TS's `flushPromises()`: under parallel load a
            # spawned task may still be pending, `task.done()` would be False, and a real
            # unhandled exception would go unrecorded -- a false negative in this guard.
            if spawned:
                await asyncio.wait(spawned, timeout=5.0)
            for task in spawned:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    captured.append(task.exception())
            captured.extend(warning.message for warning in raised if "was never awaited" in str(warning.message))


async def test_tool_updates_after_the_tool_settles_are_ignored():
    captured: list = []
    events: list = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        captured.append(on_update)
        if on_update:
            on_update(AgentToolResult(content=[TextContent(text="running")], details={}))
        return AgentToolResult(content=[TextContent(text="ok")], details={}, terminate=True)

    agent = make_agent(
        [tool_call_response(ToolCall(id="call-1", name="echo", arguments={}))],
        initial_state=MutableAgentState(model=TEST_MODEL),
    )
    agent.state.tools = [echo_tool(execute)]
    agent.subscribe(lambda event, signal: events.append(event))

    async with capture_unhandled_async_errors() as unhandled:
        await asyncio.wait_for(agent.prompt("run tool"), timeout=TIMEOUT)
        event_count_after_prompt = len(events)

        late_update = captured[0]
        assert late_update is not None
        late_update(AgentToolResult(content=[TextContent(text="late")], details={}))
        await asyncio.sleep(0)

        assert len([event for event in events if event.type == "tool_execution_update"]) == 1
        assert len(events) == event_count_after_prompt

    # Mirrors `expect(unhandledRejections).toEqual([])`: dropping the late update
    # must not leave an orphaned task or an un-awaited coroutine behind.
    assert unhandled == []


async def test_settled_parallel_tool_updates_are_ignored_while_another_tool_runs():
    slow_started = asyncio.Event()
    settled_tool_ended = asyncio.Event()
    release_slow = asyncio.Event()
    captured: list = []
    events: list = []

    async def settled_execute(tool_call_id, params, signal=None, on_update=None):
        captured.append(on_update)
        return AgentToolResult(content=[TextContent(text="done")], details={}, terminate=True)

    async def slow_execute(tool_call_id, params, signal=None, on_update=None):
        slow_started.set()
        await release_slow.wait()
        return AgentToolResult(content=[TextContent(text="done")], details={}, terminate=True)

    settled_tool = AgentTool(
        name="settled_tool",
        description="Captures progress callbacks",
        parameters={"type": "object", "properties": {}},
        label="Settled Tool",
        execute=settled_execute,
    )
    slow_tool = AgentTool(
        name="slow_tool",
        description="Keeps the agent run active",
        parameters={"type": "object", "properties": {}},
        label="Slow Tool",
        execute=slow_execute,
    )

    message = make_assistant_message(
        [
            ToolCall(id="call-1", name="settled_tool", arguments={}),
            ToolCall(id="call-2", name="slow_tool", arguments={}),
        ],
        stop_reason="toolUse",
    )
    agent = make_agent([message], initial_state=MutableAgentState(model=TEST_MODEL))
    agent.state.tools = [settled_tool, slow_tool]

    def listener(event, signal) -> None:
        events.append(event)
        if event.type == "tool_execution_end" and event.tool_call_id == "call-1":
            settled_tool_ended.set()

    agent.subscribe(listener)

    prompt_task = asyncio.ensure_future(agent.prompt("run tools"))
    await asyncio.wait_for(slow_started.wait(), timeout=TIMEOUT)
    await asyncio.wait_for(settled_tool_ended.wait(), timeout=TIMEOUT)
    event_count_before_late_update = len(events)

    late_update = captured[0]
    assert late_update is not None
    late_update(AgentToolResult(content=[TextContent(text="late")], details={}))
    await asyncio.sleep(0)
    assert len(events) == event_count_before_late_update

    release_slow.set()
    await asyncio.wait_for(prompt_task, timeout=TIMEOUT)
    assert [event for event in events if event.type == "tool_execution_update"] == []
