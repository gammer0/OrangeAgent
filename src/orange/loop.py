"""Orchestration layer: the ReAct state machine and step gating.

See docs/01-architecture.md §4.2, §4.6.1, §4.9.

This module owns **control flow only**; it executes no tools and does not build
``messages`` directly:

* :class:`StepController` -- step gating, injection counting, termination
  decisions (including the deadlock criterion).
* :class:`AgentLoop` -- wires "LLM call -> tool batch -> safe-point injection ->
  next step" together. Every dependency is injected through constructor
  arguments, so unit tests can use fakes.

Safe-point semantics (G1-G4 / D7 / D8):

========================  =================  ====================
Safe point                chat               control
========================  =================  ====================
① after the LLM returns    drain all          take first, clear rest
② between groups           **untouched**      take first, clear rest
  (more groups remain)                       -> stop remaining + fill 3e
③ whole batch done         drain all          take first, clear rest
========================  =================  ====================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from orange.exceptions import MessageInvariantError
from orange.execution import BatchPlan, ExecResult, fill_placeholders, plan_batch
from orange.executor import Committed
from orange.inbox import Inbox, InterruptRequest
from orange.jev import JevRouter
from orange.message import (
    AssistantMsg,
    Message,
    SystemMsg,
    ToolCall,
    ToolMsg,
    UserMsg,
    validate_messages,
)
from orange.routing import RouteDecision

__all__ = [
    "DeadlockTracker",
    "StepController",
    "AgentLoop",
    "LoopHooks",
    "LoopResult",
    "MAX_STEPS",
    "MAX_INJECTS",
]


#: Design-doc config item ``MAX_STEPS``.
#: Note: 20 is not enough -- we measured multi-step engineering tasks (edit code
#: + run tests + fix errors) frequently exhausting the budget before wrapping up.
MAX_STEPS = 40
#: Design-doc config item ``MAX_INJECTS``.
MAX_INJECTS = 10
#: Consecutive failures reaching this value without changing tool path counts as a
#: deadlock (design doc §4.9).
DEADLOCK_THRESHOLD = 3
#: Consecutive empty replies (no text, no tool calls) reaching this count aborts.
#: Note: an empty reply can neither be accepted as a final answer (it would
#: silently truncate the task) nor allowed to spin.
EMPTY_RESPONSE_THRESHOLD = 2


class ModelClient(Protocol):
    """Model-call protocol: the orchestration layer depends on this method alone."""

    def chat(self, messages: Sequence[Message]) -> AssistantMsg:
        """Return the next ``AssistantMsg`` (with ``tool_calls`` or final text)."""
        ...


class ToolResolver(Protocol):
    """Tool-resolution protocol: ``ToolCall -> executable definition``."""

    def __call__(self, call: ToolCall) -> Any:
        """Return an object exposing ``fn`` / ``concurrency`` / ``idempotent``."""
        ...


class BatchExecutor(Protocol):
    """Batch-execution protocol: same signature as :func:`orange.executor.execute_batch`."""

    def __call__(
        self,
        plan: BatchPlan,
        resolver: Callable[[ToolCall], Any],
        on_group_done: Callable[[int], Any] | None = None,
        *,
        max_workers: int = 4,
    ) -> dict[str, ExecResult]:
        ...


@dataclass
class DeadlockTracker:
    """Deadlock criterion: 3 consecutive failures + unchanged result hash + same tool path.

    All three must hold to declare a deadlock; breaking any one resets the count.
    """

    threshold: int = DEADLOCK_THRESHOLD
    _count: int = 0
    _last_hash: str = ""
    _last_path: str = ""

    def record(self, *, ok: bool, content: str, path: str) -> bool:
        """Record one call result; returns whether a deadlock is declared."""
        if ok:
            self.reset()
            return False

        digest = f"{path}|{hash(content)}"
        if digest == self._last_hash and path == self._last_path:
            self._count += 1
        else:
            # Tool path changed or the result changed: restart the count
            self._count = 1
            self._last_hash = digest
            self._last_path = path
        return self._count >= self.threshold

    def reset(self) -> None:
        """Reset the counters (called on success, tool switch or a new batch)."""
        self._count = 0
        self._last_hash = ""
        self._last_path = ""


class StepController:
    """Step gating, injection counting and termination decisions (§4.2).

    The three termination conditions are independent; any one of them requires
    ending the current session turn:

    * steps reach ``MAX_STEPS``;
    * injections reach ``MAX_INJECTS``;
    * the deadlock criterion fires.
    """

    def __init__(self, max_steps: int = MAX_STEPS, max_injects: int = MAX_INJECTS) -> None:
        self.max_steps = max_steps
        self.max_injects = max_injects
        self.step = 0
        self.injects = 0
        self.deadlock = DeadlockTracker()
        self._stop_reason: str | None = None

    # ------------------------------------------------------------- gates

    def begin_step(self) -> bool:
        """Begin a new step; ``False`` means the step limit is reached and work must stop."""
        if self.step >= self.max_steps:
            self._stop_reason = self._stop_reason or "max_steps"
            return False
        self.step += 1
        return True

    def can_inject(self) -> bool:
        """Whether another user message may be injected in this turn."""
        return self.injects < self.max_injects

    def record_inject(self) -> bool:
        """Record one injection; ``False`` means the limit was reached by this one."""
        self.injects += 1
        if self.injects >= self.max_injects:
            self._stop_reason = self._stop_reason or "max_injects"
            return False
        return True

    def record_tool_result(self, *, ok: bool, content: str, path: str) -> bool:
        """Record one tool result; returns whether the deadlock criterion fired."""
        if self.deadlock.record(ok=ok, content=content, path=path):
            self._stop_reason = self._stop_reason or "deadlock"
            return True
        return False

    # ------------------------------------------------------- termination

    @property
    def stop_reason(self) -> str | None:
        """Stop reason for this session.

        One of ``max_steps`` / ``max_injects`` / ``deadlock`` / ``None``.
        """
        return self._stop_reason

    @property
    def should_stop(self) -> bool:
        """Whether the session should end."""
        return self._stop_reason is not None

    def mark_stop(self, reason: str) -> None:
        """Record a reason when an external condition (context wrap-up, ...) requires stopping."""
        self._stop_reason = self._stop_reason or reason


@dataclass
class LoopHooks:
    """Optional callbacks during orchestration; all default to no-ops.

    Attributes:
        on_assistant: Called after a model reply (for persistence / tracing).
        on_group_done: Called after each ``ExecGroup`` completes and is backfilled.
        on_batch_done: Called once a whole tool batch has been backfilled into the
            message list, so callers can observe the tool results themselves.
        on_inject: Called after each injected user message.
        on_abort: Called when an abort happens (control signal); the argument is
            the reason.
        on_summary: Called when context compression fires.
        on_descent: Called with a descent's raw output when its turn is dropped
            from history, so the caller can move the children into the
            discovered-tools inventory.
        on_delta: Called with each text fragment as the model produces it, so a
            UI can render the answer while it is still being generated. This is a
            preview only -- the complete reply arrives as one ``AssistantMsg``.
    """

    on_assistant: Callable[[AssistantMsg], None] | None = None
    on_group_done: Callable[[int], None] | None = None
    on_batch_done: Callable[[Sequence[Message]], None] | None = None
    on_inject: Callable[[UserMsg, str], None] | None = None
    on_abort: Callable[[str], None] | None = None
    on_summary: Callable[[str], None] | None = None
    on_descent: Callable[[str], None] | None = None
    on_delta: Callable[[str], None] | None = None


@dataclass
class LoopResult:
    """Final result of one session turn.

    Attributes:
        messages: Session messages (not persisted; the caller decides).
        final_text: Final answer text; ``None`` when ending due to an abort.
        steps: Number of steps actually taken.
        stop_reason: Reason for stopping.
        aborted: Whether the turn ended due to a ``control`` signal.
    """

    messages: list[Message]
    final_text: str | None
    steps: int
    stop_reason: str | None
    aborted: bool = False
    trace: list[str] = field(default_factory=list)
    route: RouteDecision | None = None


class AgentLoop:
    """The ReAct main loop (§4.2).

    Responsibility boundaries:

    * **Is** the sole owner of control flow: decides when to call the model,
      when to execute tools and when to check safe points.
    * **Is not** a tool executor: batch execution is delegated to the injected
      ``executor``.
    * **Is not** a message builder: the ``SystemMsg`` and projection belong to the
      context layer; this class only appends ``UserMsg`` / ``AssistantMsg`` /
      ``ToolMsg``.

    Every constructor argument is injected, so unit tests can fully cover the
    safe-point semantics with a fake model and a fake executor.
    """

    def __init__(
        self,
        *,
        model: ModelClient,
        inbox: Inbox,
        executor: BatchExecutor,
        resolver: Callable[[ToolCall], Any],
        system_msg: SystemMsg,
        controller: StepController | None = None,
        hooks: LoopHooks | None = None,
        max_workers: int = 4,
        assembler: Any | None = None,
        router: JevRouter | None = None,
        branch_provider: Callable[[], Sequence[str]] | None = None,
        tool_count_provider: Callable[[], int] | None = None,
        history_tokens_provider: Callable[[int], int] | None = None,
    ) -> None:
        self.model = model
        self.inbox = inbox
        self.executor = executor
        self.resolver = resolver
        self.system_msg = system_msg
        self.controller = controller or StepController()
        self.hooks = hooks or LoopHooks()
        self.max_workers = max_workers
        #: Optional context assembler; when absent the model is called directly
        #: with a naive path (lossless degradation per I4).
        self.assembler = assembler
        #: Consecutive empty replies (no text, no tool calls); see
        #: EMPTY_RESPONSE_THRESHOLD.
        self._empty_responses = 0
        self.empty_response_threshold = EMPTY_RESPONSE_THRESHOLD
        #: Optional Jev router; when absent no routing happens (equivalent to full
        #: exposure).
        self.router = router
        #: Candidate top-level branch provider; used together with ``router``.
        self.branch_provider = branch_provider
        #: Total tool count provider, for the routing enable conditions.
        self.tool_count_provider = tool_count_provider
        #: History token provider, for the routing enable conditions.
        self.history_tokens_provider = history_tokens_provider
        #: Intermediate (abstract) paths the caller considers tree descents. A call
        #: to one of these is a discovery step, not conversation, so its turn is
        #: omitted from the projected request (see ``visible_history``).
        self.descent_paths: frozenset[str] = frozenset()
        #: Tool-call ids that belong to descent turns; populated as they happen.
        self.descent_call_ids: set[str] = set()
        #: Descents issued during **this run**. They stay visible to the model; only
        #: older ones are dropped from the projected request.
        self.current_turn_ids: set[str] = set()
        #: Most recent routing decision (so callers can write traces).
        self.last_route: RouteDecision | None = None

    # ------------------------------------------------------------ main loop

    def run(self, history: Sequence[Message] | None = None) -> LoopResult:
        """Run until a final answer, step exhaustion or an abort.

        Args:
            history: Existing session messages; ``None`` means a new session
                (the ``SystemMsg`` is assembled automatically).

        Returns:
            A :class:`LoopResult` with the final text and stop reason.
        """
        messages: list[Message] = list(history) if history else [self.system_msg]
        if not isinstance(messages[0], SystemMsg):
            raise MessageInvariantError("A1", "messages[0] must be SystemMsg")

        aborted = False
        final_text: str | None = None

        while self.controller.begin_step():
            # ---- Safe point 1: consume injections left over from the previous
            # turn before calling the LLM ----
            if not self._drain_at_safe_point(messages, "safe_point_1"):
                aborted = True
                break

            validate_messages(messages)

            outgoing = self._assemble(messages)
            # Earlier turns' descents are omitted: their tools are already in the
            # discovered-tools inventory. The current turn's descents stay, since
            # the model needs those children to pick its next call.
            outgoing = self.visible_history(outgoing)
            assistant = self._call_model(outgoing)
            messages.append(assistant)
            if self.hooks.on_assistant is not None:
                self.hooks.on_assistant(assistant)

            if assistant.is_final:
                final_text = assistant.content
                break

            # ---- Empty reply: neither tool_calls nor text ----
            # Note: the model occasionally returns content="" with no calls. That
            # can neither be a final answer (it would silently truncate the task)
            # nor proceed to plan_batch (an empty plan spins). Count a failure and
            # abort after empty_response_threshold consecutive ones.
            if not assistant.tool_calls:
                self._empty_responses += 1
                if self._empty_responses >= self.empty_response_threshold:
                    self.controller.mark_stop("empty_response")
                    break
                continue
            self._empty_responses = 0

            # ---- Jev routing (an optimization; fall back to full on failure, I4) ----
            self._maybe_route(messages)

            # ---- Tool batch ----
            plan = plan_batch(list(assistant.tool_calls), self._concurrency_of)
            stop_after = self._execute(plan, messages)

            # Notify listeners once the whole batch has been backfilled into
            # ``messages``. ``on_group_done`` fires before the results exist, so it
            # cannot report them; this hook can.
            if self.hooks.on_batch_done is not None:
                self.hooks.on_batch_done(messages)

            # ---- Tree descent cleanup ----
            # Descents are a means to an end: once a **concrete** tool has been
            # called, the chain that led to it can be dropped from the model's
            # context. Dropping it earlier would hide the children the model just
            # asked for and make it repeat the same descent forever, so cleanup is
            # gated on a real (non-descent) call existing in this batch.
            self._mark_descent_turns(messages, plan)
            self._prune_resolved_descents(messages, plan)

            if stop_after is not None:
                # D7 / G1: fill placeholders after stopping the remaining groups,
                # then insert the attached text per G3.
                self._fill_and_append(messages, plan, stop_after)
                aborted = True
                if self.hooks.on_abort is not None:
                    self.hooks.on_abort(stop_after.reason)
                break

            # ---- Safe point 3: batch complete, both channels consumable ----
            if not self._drain_at_safe_point(messages, "safe_point_3"):
                aborted = True
                break

            if self.controller.should_stop:
                break

        if self.controller.stop_reason is not None and self.controller.stop_reason != "max_steps":
            aborted = aborted or self.controller.stop_reason == "deadlock"

        return LoopResult(
            messages=messages,
            final_text=final_text,
            steps=self.controller.step,
            stop_reason=self.controller.stop_reason,
            aborted=aborted,
            route=self.last_route,
        )

    # --------------------------------------------------- context assembly

    def _assemble(self, messages: list[Message]) -> list[Message]:
        """Assemble the messages to send (C1/C2/C3).

        Without an injected ``assembler`` the list is returned as-is -- a missing
        optimization must degrade losslessly (I4). When the watermark reaches
        ``session_end`` a wrap-up reason is recorded and this turn stops advancing.
        """
        if self.assembler is None:
            return messages

        result = self.assembler.assemble(messages)
        if result.compressed:
            # Compression rebuilds the list, so we must continue with the new one;
            # otherwise later appends would write into the old object.
            messages[:] = result.messages
            if self.hooks.on_summary is not None:
                self.hooks.on_summary(result.action)

        if result.action == "session_end":
            self.controller.mark_stop("session_end")
        return result.messages

    # -------------------------------------------------------------- routing

    def _maybe_route(self, messages: list[Message]) -> None:
        """Run one Jev routing pass when the enable conditions hold.

        ``state`` carries the query only (the last ``UserMsg``) and never any tool
        output; every exception converges to a fallback decision (I4).
        """
        if self.router is None or self.branch_provider is None:
            return

        branches = list(self.branch_provider())
        if not branches:
            return

        query = _last_user_text(messages)
        history_tokens = (
            self.history_tokens_provider(len(messages))
            if self.history_tokens_provider is not None
            else 0
        )
        tool_count = self.tool_count_provider() if self.tool_count_provider is not None else 0

        self.last_route = self.router.route(
            query,
            branches,
            history_tokens=history_tokens,
            tool_count=tool_count,
        )

    # ------------------------------------------------------ descent cleanup

    def _mark_descent_turns(self, messages: list[Message], plan: BatchPlan) -> None:
        """Record descent turns and hand their children to the inventory.

        A descent is a call to an abstract node. Its output lists that node's
        children, which are exactly what the discovered-tools inventory carries,
        so the turn is redundant **for the model** from the next step onward.

        The turn is deliberately *not* removed from ``messages``. Removing it
        would hide the result the model just asked for, and it would then repeat
        the same descent forever (observed: eight identical ``/tool`` calls until
        ``max_steps``). Instead the ids are recorded in :attr:`descent_call_ids`
        and the projection layer omits those turns when building the outgoing
        request -- so history stays valid and complete while the model still sees
        a clean transcript.
        """
        if not plan.groups:
            return

        for group in plan.groups:
            for call in group.calls:
                if self._is_descent(call.name):
                    self.descent_call_ids.add(call.id)
                    self.current_turn_ids.add(call.id)

        if not self.descent_call_ids:
            return

        # Feed the children into the inventory while the results are available.
        if self.hooks.on_descent is not None:
            for message in messages:
                if (
                    isinstance(message, ToolMsg)
                    and message.tool_call_id in self.descent_call_ids
                ):
                    self.hooks.on_descent(message.content)

    def _call_model(self, outgoing: Sequence[Message]) -> AssistantMsg:
        """Call the model, forwarding text fragments to the UI when supported.

        Streaming is optional by design: the orchestration layer talks to a
        ``chat(messages)`` callable, and only the web runner's client exposes a
        ``chat_stream`` variant. When it is absent (CLI, tests) the plain call is
        used, so behaviour there is unchanged.

        The streaming callback is a *preview* channel only -- the assistant message
        appended to history is always the single complete reply, never the
        fragments.
        """
        chat_stream = getattr(self.model, "chat_stream", None)
        if chat_stream is None:
            return self.model.chat(outgoing)

        emitted = [False]

        def on_delta(text: str) -> None:
            if not text:
                return
            emitted[0] = True
            if self.hooks.on_delta is not None:
                self.hooks.on_delta(text)

        try:
            return chat_stream(outgoing, on_delta=on_delta)
        except TypeError:
            # A model object that defines ``chat_stream`` with a different
            # signature must not break the turn; fall back to the plain call.
            if emitted[0]:
                raise
            return self.model.chat(outgoing)

    def _is_descent(self, path: str) -> bool:
        """Whether ``path`` names an abstract node rather than a real tool.

        The loop has no direct view of the tree, so the caller supplies the set
        of intermediate paths through ``descent_paths``.
        """
        return path in self.descent_paths

    def _prune_resolved_descents(
        self, messages: list[Message], plan: BatchPlan
    ) -> None:
        """Drop descent turns once a concrete tool has been called.

        This is the "clean up as soon as the choice is made" rule. The gate is
        deliberate: pruning is only allowed when **this batch contains at least
        one real tool call**. A batch made purely of descents is still mid-search,
        and removing its results would leave the model without the children it
        just asked for -- it would then descend the same path again, forever.

        The tool these descents revealed is already in the discovered-tools
        inventory, which is injected on every request, so the model keeps the
        knowledge without the raw JSON staying in the transcript.
        """
        if not plan.groups:
            return

        real_calls = [
            call
            for group in plan.groups
            for call in group.calls
            if not self._is_descent(call.name)
        ]
        if not real_calls:
            # Pure discovery batch: keep everything so the model can act next.
            return

        # Every descent resolved so far belongs to the chain that produced this
        # call (or to earlier chains, which are equally obsolete). Drop them all.
        self._prune_descents(messages, self.descent_call_ids)

    def _prune_descents(self, messages: list[Message], doomed: set[str]) -> None:
        """Remove the given descent calls and their results from ``messages``."""
        if not doomed:
            return

        kept: list[Message] = []
        for message in messages:
            if isinstance(message, AssistantMsg):
                remaining = [c for c in message.tool_calls if c.id not in doomed]
                if message.tool_calls and not remaining:
                    continue
                if len(remaining) != len(message.tool_calls):
                    message = AssistantMsg(content=message.content, tool_calls=remaining)
            elif isinstance(message, ToolMsg) and message.tool_call_id in doomed:
                continue
            kept.append(message)
        messages[:] = kept

    def visible_history(self, messages: Sequence[Message]) -> list[Message]:
        """Project history for a request, omitting unresolved descent turns.

        Live turns are already pruned in place the moment a concrete tool is
        chosen (see :meth:`_prune_resolved_descents`). This projection exists for
        history that arrived from storage: a restored session still contains its
        descent turns -- the user's trace must show how a tool was found -- while
        the model does not need to re-read them, since the tools they revealed are
        in the discovered-tools inventory.

        Descents issued during **this** run are kept: the model is still using
        those children to choose its next call.
        """
        drop = self.descent_call_ids - self.current_turn_ids
        if not drop:
            return list(messages)

        kept = list(messages)
        self._prune_descents(kept, drop)
        return kept

    # ------------------------------------------------------ batch execution

    def _concurrency_of(self, call: ToolCall) -> str:
        """Read concurrency from ``ToolDef``; fall back conservatively to ``exclusive`` (I4)."""
        try:
            tool = self.resolver(call)
        except Exception:
            return "exclusive"
        return getattr(tool, "concurrency", "exclusive")

    def _execute(
        self,
        plan: BatchPlan,
        messages: list[Message],
    ) -> InterruptRequest | None:
        """Execute the whole batch; a returned interrupt means the remaining groups must stop.

        The group-done callback implements safe point ②: it consumes ``control``
        only and **never** consumes ``chat`` (G2).
        """
        interrupt: InterruptRequest | None = None

        def on_group_done(report: Any) -> Any:
            nonlocal interrupt
            if self.hooks.on_group_done is not None:
                self.hooks.on_group_done(getattr(report, "index", 0))
            signal = self.inbox.take_control_at_sub_batch()
            if signal is not None:
                interrupt = signal
                return Committed(reason=signal.reason)
            return None

        results = self.executor(
            plan,
            self.resolver,
            on_group_done,
            max_workers=self.max_workers,
        )

        # D3: backfill in group order then intra-group order, never completion order
        for group in plan.groups:
            for call in group.calls:
                result = results.get(call.id)
                if result is None:
                    continue
                messages.append(
                    ToolMsg(
                        tool_call_id=call.id,
                        name=call.name,
                        content=result.content,
                    )
                )
                if self.controller.record_tool_result(
                    ok=result.ok, content=result.content, path=call.name
                ):
                    # Deadlock criterion fired: stop advancing
                    self.controller.mark_stop("deadlock")

        return interrupt

    def _fill_and_append(
        self,
        messages: list[Message],
        plan: BatchPlan,
        signal: InterruptRequest,
    ) -> None:
        """G1 / D7: fill ``3e`` placeholders.

        G3: insert a ``UserMsg`` when ``text`` is non-empty.
        """
        filled = {
            msg.tool_call_id
            for msg in messages
            if isinstance(msg, ToolMsg) and msg.tool_call_id
        }
        placeholders = fill_placeholders(plan, filled, signal.reason)
        messages.extend(placeholders)

        if signal.text:
            self._inject_user_msg(messages, signal.text, channel="control")

    # ------------------------------------------------------------ safe points

    def _drain_at_safe_point(self, messages: list[Message], label: str) -> bool:
        """Full safe point: handle ``control`` first, then ``chat`` (G4).

        Returns:
            ``True`` to continue; ``False`` when a ``control`` signal arrived and
            this turn should end.
        """
        signal, chats = self.inbox.drain_at_full_safe_point()
        if signal is not None:
            # At ① / ③ no remaining groups exist to stop, so this becomes the
            # abort reason directly.
            return False

        for item in chats:
            self._inject_user_msg(messages, item.text, channel="chat")
        del label
        return True

    def _inject_user_msg(self, messages: list[Message], text: str, channel: str) -> None:
        """Insert one ``UserMsg`` and record the injection count.

        Insertion inside an unclosed ``tool_calls`` sequence is forbidden
        (D8 / I6) -- safe points guarantee callers only invoke this at legal
        positions; this is a defensive check.
        """
        if _has_unclosed_tool_calls(messages):
            raise MessageInvariantError(
                "I6",
                f"{channel} channel messages must not be inserted into an "
                "unclosed tool_calls sequence",
            )

        msg = UserMsg(content=text)
        messages.append(msg)
        if self.hooks.on_inject is not None:
            self.hooks.on_inject(msg, channel)
        if not self.controller.record_inject():
            self.controller.mark_stop("max_injects")


def _has_unclosed_tool_calls(messages: Sequence[Message]) -> bool:
    """Whether any ``tool_calls`` remain unpaired."""
    pending: set[str] = set()
    for msg in messages:
        if isinstance(msg, AssistantMsg):
            pending.update(call.id for call in msg.tool_calls)
        elif isinstance(msg, ToolMsg):
            pending.discard(msg.tool_call_id)
    return bool(pending)


def _last_user_text(messages: Sequence[Message]) -> str:
    """Return the text of the most recent ``UserMsg``, used as the routing query.

    Design doc §4.4: ``state`` carries the query only, never tool output (guards
    against JPDH).
    """
    for msg in reversed(messages):
        if isinstance(msg, UserMsg):
            return msg.content
    return ""
