from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from .envs.base import Task
from .envs.registry import ENVIRONMENTS, METHODS
from .runtime import Episode, RunConfig, run


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_lines(path, rows):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()


def provider(configuration, name):
    value = configuration.get("providers", {}).get(name)
    if not isinstance(value, dict) or not value.get("factory"):
        raise ValueError(f"Configure providers.{name}.factory as module:factory")
    module, separator, attribute = value["factory"].partition(":")
    if not separator or not module or not attribute:
        raise ValueError("Provider factory must use module:factory")
    factory = getattr(importlib.import_module(module), attribute)
    return factory(dict(value.get("options", {})))


def main():
    parser = argparse.ArgumentParser(prog="conpact-agent")
    commands = parser.add_subparsers(dest="command", required=True)
    rollout = commands.add_parser("run")
    rollout.add_argument("--config", required=True)
    rollout.add_argument("--tasks", required=True)
    rollout.add_argument("--out", required=True)
    rollout.add_argument("--environment", choices=ENVIRONMENTS)
    rollout.add_argument("--method", choices=METHODS)
    rollout.add_argument("--rollouts", type=int, default=8)
    dataset = commands.add_parser("data")
    dataset.add_argument(
        "--method",
        choices=[
            "conpact_s",
            "conpact_r",
            "react",
            "plan_act",
            "paa",
            "hsl",
            "webstar",
            "ecot",
        ],
        required=True,
    )
    dataset.add_argument("--input", required=True)
    dataset.add_argument("--out", required=True)
    dataset.add_argument("--config")
    dataset.add_argument(
        "--include-uncertain", action="store_true",
        help="ConPAct-S/ConPAct-R only: include separately labeled uncertainty targets after reviewer approval",
    )
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--input", required=True)
    args = parser.parse_args()
    if (
        args.command == "data" and args.include_uncertain
        and args.method not in ("conpact_s", "conpact_r")
    ):
        parser.error("--include-uncertain is available only for ConPAct-S and ConPAct-R")
    configuration = read_json(args.config) if getattr(args, "config", None) else {}
    if args.command == "run":
        if args.rollouts < 1:
            parser.error("--rollouts must be positive")
        parameters = dict(configuration.get("run", {}))
        if args.method:
            parameters["method"] = args.method
        settings = RunConfig(**parameters)
        environment = args.environment or configuration.get("environment")
        if environment not in ENVIRONMENTS:
            parser.error("Set an environment")
        planner = provider(configuration, "planner")
        actor = (
            provider(configuration, "actor")
            if configuration.get("providers", {}).get("actor", {}).get("factory")
            else planner
        )
        tasks = [Task(**value) for value in read_json(args.tasks)]
        if not tasks or len({task.id for task in tasks}) != len(tasks):
            raise ValueError("Task IDs must be nonempty and unique")

        def episodes():
            for task in tasks:
                for index in range(args.rollouts):
                    sandbox = provider(configuration, "sandbox")
                    yield run(
                        environment, task, planner, sandbox, settings, actor, index
                    ).to_dict()

        write_lines(args.out, episodes())
    elif args.command == "data":
        from .data import conpact_s, conpact_r
        from .baselines import registry as baselines

        episodes = [Episode.from_dict(value) for value in read_lines(args.input)]
        if args.method == "conpact_s":
            reviewer = provider(configuration, "reviewer") if args.include_uncertain else None
            rows = conpact_s.build(
                episodes, include_uncertain=args.include_uncertain, reviewer=reviewer,
            )
        elif args.method in ("react", "plan_act"):
            rows = baselines.build(args.method, episodes)
        else:
            teacher = provider(configuration, "teacher")
            if args.method in ("paa", "ecot", "conpact_r"):
                reviewer = provider(configuration, "reviewer")
            if args.method == "conpact_r":
                verifier = provider(configuration, "verifier")
                rows = [
                    row
                    for episode in episodes
                    for row in conpact_r.reconstruct(
                        episode, teacher, reviewer, verifier,
                        include_uncertain=args.include_uncertain,
                    )[
                        "records"
                    ]
                ]
            elif args.method == "paa":
                rows = baselines.build(
                    "paa", episodes, teacher=teacher, reviewer=reviewer
                )
            elif args.method == "ecot":
                rows = baselines.build(
                    "ecot", episodes, teacher=teacher, reviewer=reviewer
                )
            elif args.method == "hsl":
                rows = baselines.build("hsl", episodes, teacher=teacher)
            else:
                rows = baselines.build("webstar", episodes, teacher=teacher)
        write_lines(args.out, rows)
        print(json.dumps({"records": len(rows)}))
    else:
        from .metrics import summarize

        print(
            json.dumps(
                summarize(
                    [Episode.from_dict(value) for value in read_lines(args.input)]
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
