from __future__ import annotations

import copy
import hashlib
import json

from ..envs.registry import get
from ..methods.messages import _image, _text


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def sample(episode, turn, *, method=None):
    if turn.get("error") or not turn.get("raw"):
        raise ValueError("Cannot supervise a failed model call")
    messages = copy.deepcopy(turn["messages"])
    messages.append({"role": "assistant", "content": turn["raw"]})
    return {
        "messages": messages,
        "loss": [0] * (len(messages) - 1) + [1],
        "meta": {
            "environment": episode.environment,
            "task_id": episode.task.id,
            "rollout_id": episode.rollout_id,
            "role": turn["role"],
            "step": turn["step"],
            "turn": turn["turn"],
            "method": method or episode.method,
            "source": digest(
                {
                    "task": episode.task.id,
                    "actions": episode.actions,
                    "frames": [f.image_b64 for f in episode.frames],
                }
            ),
        },
    }


def public_frame(episode, index):
    frame = episode.frames[index]
    environment = get(episode.environment, episode.task)
    prompt = environment.prompt("react", frame)
    return [
        _text(f"Observation {index}. " + prompt.instruction),
        _image(frame.image_b64),
    ]


def source_messages(episode, instruction, *, start=0, stop=None):
    stop = len(episode.actions) if stop is None else stop
    if not 0 <= start <= stop <= len(episode.actions):
        raise ValueError("Invalid observation interval")
    env = get(episode.environment, episode.task)
    prompt = env.prompt("react", episode.frames[start])
    content = [_text("Task and environment rules:\n" + prompt.actor_system)]
    for index in range(start, stop + 1):
        if episode.frames[index].metadata.get("terminal_screenshot_may_be_autoreset"):
            content.append(
                _text(
                    f"Observation {index} is unavailable after environment termination."
                )
            )
        else:
            content.extend(public_frame(episode, index))
        if index < stop:
            content.append(_text("Executed action: " + episode.actions[index]))
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": content},
    ]


def ask_json(client, messages):
    raw = client.complete(messages)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (TypeError, ValueError) as exc:
        raise ValueError("Teacher must return one valid JSON value") from exc


def review_sample(client, record):
    content = [_text("Review this exact student input and candidate reply:")]
    for message in record["messages"]:
        content.append(_text("MESSAGE ROLE: " + message["role"]))
        if isinstance(message["content"], str):
            content.append(_text(message["content"]))
        else:
            content.extend(copy.deepcopy(message["content"]))
    value = ask_json(
        client,
        [
            {
                "role": "system",
                "content": "Independently review the candidate using ONLY the student input shown here. "
                "Reject incorrect current-state claims, unjustified subgoals/actions, unsupported corrections, "
                "incorrect handback, and any claim requiring a future or unavailable observation. "
                "Earlier discussion may contain intentionally incorrect assertions; judge the final reply independently. "
                "A schema-permitted unknown/null is a valid abstention only when the shown observation "
                "does not support a definite reading. Reject unsupported abstention or guesses. "
                "Agreement or absence of conflict alone does not establish correctness. "
                "A candidate action has not yet executed. Return exactly JSON with accept:boolean and reason:string.",
            },
            {"role": "user", "content": content},
        ],
    )
    if (
        not isinstance(value, dict)
        or type(value.get("accept")) is not bool
        or not isinstance(value.get("reason"), str)
        or not value["reason"]
    ):
        raise ValueError("Invalid independent review")
    return value
