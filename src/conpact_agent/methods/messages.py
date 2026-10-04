from __future__ import annotations
from conpact_agent.envs.base import PromptSet, fill


def _text(value: str) -> dict:
    return {"type": "text", "text": value}


def _image(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


_PNG_BASE64_PREFIX = "iVBORw0KGgo"


def _board(prompts: PromptSet, board: str) -> dict:
    if prompts.obs == "text":
        if not board:
            raise RuntimeError(
                "text arm was given an empty board; the environment produced no "
                "textual observation"
            )

        if board.startswith(_PNG_BASE64_PREFIX):
            raise RuntimeError(
                "text arm was given a base64 PNG instead of a board; the caller "
                "read the frames as images. Text frames are replayed with "
                "provide a textual observation for a text prompt"
            )
        return _text("```\n" + board + "\n```")
    return _image(board)


def actor_messages(
    prompts: PromptSet,
    subgoal: str,
    frames: list[str],
    actions: list[str],
    window: int,
    extra: dict | None = None,
    locator: str | None = None,
    context: str | None = None,
) -> list:

    done = len(actions)
    kept = max(0, min(done, window - 1))
    start = done - kept

    text_history = prompts.history_mode == "text"
    has_history = done > 0 if text_history else kept > 0
    content = [
        _text(prompts.instruction_with_history if has_history else prompts.instruction)
    ]
    for j in range(0 if text_history else start, done):
        if j >= start:
            content.append(_board(prompts, frames[j]))
        content.append(_text(f"\n<action>{actions[j]}</action>\n"))
    content.append(_board(prompts, frames[done]))
    if context is not None:
        content.append(_text(context))
    if locator is not None:
        content.append(_text(locator))
    system = fill(
        prompts.actor_system,
        subgoal=subgoal or prompts.subgoal_fallback,
        **(extra or {}),
    )
    return [
        {"role": "system", "content": [_text(system)]},
        {"role": "user", "content": content},
    ]


def plan_messages(prompts: PromptSet, frame: str) -> list:

    return [
        {"role": "system", "content": [_text(prompts.planner_system)]},
        {
            "role": "user",
            "content": [
                _text(prompts.planner_instruction),
                _text(prompts.current_frame_block),
                _board(prompts, frame),
            ],
        },
    ]


def hint_messages(
    prompts: PromptSet,
    global_plan: str,
    milestone_index: int,
    milestone: str,
    frames: list[str],
    actions: list[str],
    window: int,
) -> list:

    done = len(actions)
    kept = max(0, min(done, window - 1))
    start = done - kept
    content = [
        _text(
            prompts.hint_instruction_with_history if kept else prompts.hint_instruction
        )
    ]
    for j in range(start, done):
        content.append(_board(prompts, frames[j]))
        content.append(_text(f"\n<action>{actions[j]}</action>\n"))
    content.append(_board(prompts, frames[done]))
    system = fill(
        prompts.hint_system,
        global_plan=global_plan,
        milestone_index=milestone_index,
        subgoal=milestone,
    )
    return [
        {"role": "system", "content": [_text(system)]},
        {"role": "user", "content": content},
    ]


def planner_messages(
    prompts: PromptSet,
    step: int,
    frames: list[str],
    subgoal_history: list[tuple[int, str]],
    prev_subgoal: str,
    prev_status: str,
    window: int,
    locator: str | None = None,
    context: str | None = None,
) -> list:

    content: list[dict] = []
    if subgoal_history:
        kept = min(len(subgoal_history), max(0, window - 1))
        older = subgoal_history[: len(subgoal_history) - kept]
        recent = subgoal_history[-kept:] if kept else []
        if older:
            items = "\n".join(
                fill(prompts.history_subgoal_item, step=s, subgoal=g) for s, g in older
            )
            content.append(_text(fill(prompts.history_subgoal_block, items=items)))
        content.append(_text(prompts.history_frame_block))
        for s, g in recent:
            content.append(_text(fill(prompts.history_frame_item, step=s, subgoal=g)))
            content.append(_board(prompts, frames[s]))
    if prev_subgoal:
        content.append(
            _text(
                fill(
                    prompts.planner_prev_block,
                    subgoal=prev_subgoal,
                    status=prev_status or "UNKNOWN",
                )
            )
        )
    content.append(_text(prompts.current_frame_block))
    content.append(_board(prompts, frames[step]))
    if context is not None:
        content.append(_text(context))
    if locator is not None:
        content.append(_text(locator))
    return [
        {"role": "system", "content": [_text(prompts.planner_system)]},
        {"role": "user", "content": content},
    ]
