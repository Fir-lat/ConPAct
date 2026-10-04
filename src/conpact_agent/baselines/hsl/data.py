from __future__ import annotations
import json
from ...data.records import ask_json, sample, source_messages
from ...data.selection import one_per_level, make_turn
from ...envs.registry import get
from ...methods import messages as m
import copy
import re
from collections import defaultdict
from . import protocol as hsl


def replace_objective(messages, goal):
    messages = copy.deepcopy(messages)

    def change(text):
        text = re.sub(
            r"(?m)^(#{1,3} (?:Objective|Task Objective)\n).*?(?=^#{1,3} |\Z)",
            lambda match: match.group(1) + goal + "\n",
            text,
            flags=re.S,
        )
        text = re.sub(r"(?m)^Mission:.*$", lambda _: "Mission: " + goal, text)
        return text

    for message in messages:
        if isinstance(message["content"], str):
            message["content"] = change(message["content"])
        else:
            for item in message["content"]:
                if item.get("type") == "text":
                    item["text"] = change(item["text"])
    prefix = (
        "Current task objective: "
        + goal
        + "\nThis replaces the original overall objective for this demonstration.\n"
    )
    if isinstance(messages[0]["content"], str):
        messages[0]["content"] = prefix + messages[0]["content"]
    else:
        messages[0]["content"].insert(0, m._text(prefix))
    return messages


def build(episodes, teacher):
    rows = []
    selected = one_per_level(episodes, hindsight=True)
    expert_keys = {(episode.environment, episode.task.id) for episode, _ in selected}
    for episode, cutoff in selected:
        for turn in episode.turns:
            if (
                turn["role"] == "actor"
                and turn.get("executed")
                and turn["step"] < cutoff
            ):
                rows.append(sample(episode, turn, method="hsl"))
    groups = defaultdict(list)
    for episode in episodes:
        if (
            episode.environment,
            episode.task.id,
        ) not in expert_keys and episode.end != "infrastructure_error":
            groups[(episode.environment, episode.task.id)].append(episode)
    for (name, _), sources in groups.items():
        sources.sort(key=lambda episode: episode.rollout_id)
        first = sources[0]
        if any(
            episode.frames[0].image_b64 != first.frames[0].image_b64
            for episode in sources
        ):
            raise ValueError(
                "HSL goal menus require the same initial observation across attempts"
            )
        instruction = (
            hsl.MENU_SYSTEM
            + "\nAllowed types: "
            + json.dumps(hsl.GOAL_TYPES[name])
            + "\n"
            + hsl.ENV_RULES[name]
        )
        menu = ask_json(teacher, source_messages(first, instruction, stop=0))
        hsl.validate_menu(menu, name)
        chosen = False
        for episode in sources:
            cutoff = len(episode.actions)
            if episode.frames[-1].metadata.get("terminal_screenshot_may_be_autoreset"):
                cutoff = max(0, cutoff - 1)
            achieved = ask_json(
                teacher,
                source_messages(
                    episode,
                    hsl.ACHIEVED_SYSTEM + "\nMenu: " + json.dumps(menu["goals"]),
                    stop=cutoff,
                ),
            )
            hsl.validate_achieved(achieved, menu["goals"], cutoff)
            for item in sorted(achieved["achieved"], key=lambda value: value["id"]):
                goal = next(g["text"] for g in menu["goals"] if g["id"] == item["id"])
                end = item["first_frame"]
                labels = ask_json(
                    teacher,
                    source_messages(
                        episode, hsl.RELEVANCE_SYSTEM + "\nGoal: " + goal, stop=end
                    ),
                )
                hsl.validate_relevance(labels, end)
                if labels.get("reject") or "relevant" not in labels["relevance"]:
                    continue
                env = get(name, episode.task)
                for step, label in enumerate(labels["relevance"]):
                    if label != "relevant":
                        continue
                    prompt = env.prompt("react", episode.frames[step])
                    messages = replace_objective(
                        m.actor_messages(
                            prompt,
                            "",
                            [frame.image_b64 for frame in episode.frames[: step + 1]],
                            episode.actions[:step],
                            3,
                        ),
                        goal,
                    )
                    turn = make_turn(
                        "actor",
                        step,
                        messages,
                        f"<action>{episode.actions[step]}</action>",
                        step,
                    )
                    rows.append(sample(episode, turn, method="hsl"))
                chosen = True
                break
            if chosen:
                break
    return rows
