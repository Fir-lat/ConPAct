from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy

from .envs.base import Frame, Task
from .envs.registry import Environment, get
from .interfaces import InfrastructureError, ModelError, ModelClient, Sandbox
from .methods.budget import CallBudget, CallBudgetExhausted


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class RunConfig:
    method: str = "conpact_i"
    max_steps: int | None = None
    max_calls: int | None = None
    history_frames: int = 3
    discussion_rounds: int = 3
    max_discussion_turns: int = 12
    plan_samples: int = 2
    assertion_visibility: str = "auto"
    system_intervention: bool | None = None

    def __post_init__(self):
        if self.assertion_visibility not in ("auto", "independent", "shared"):
            raise ValueError("Invalid assertion_visibility")
        if self.system_intervention is not None and type(self.system_intervention) is not bool:
            raise ValueError("system_intervention must be a boolean or null")
        for key in ("max_steps", "max_calls"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be positive")
        for key in (
            "discussion_rounds",
            "max_discussion_turns",
            "plan_samples",
            "history_frames",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < (
                0 if key == "discussion_rounds" else 1
            ):
                raise ValueError(f"Invalid {key}")


@dataclass
class Episode:
    environment: str
    task: Task
    method: str
    rollout_id: int = 0
    frames: list[Frame] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    solved: bool = False
    end: str = "action_limit"
    model_calls: int = 0
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["task"] = Task(**data["task"])
        data["frames"] = [Frame(**frame) for frame in data.get("frames", [])]
        episode = cls(**data)
        if (
            episode.end != "infrastructure_error"
            and len(episode.frames) != len(episode.actions) + 1
        ):
            raise ValueError(
                "An episode requires one more observation than executed actions"
            )
        return episode


class Session:
    def __init__(
        self,
        environment: Environment,
        task: Task,
        config: RunConfig,
        planner: ModelClient,
        actor: ModelClient,
        sandbox: Sandbox,
        rollout_id=0,
    ):
        self.environment, self.task, self.config = environment, task, config
        self.clients = {"planner": planner, "actor": actor}
        self.sandbox = sandbox
        self.budget = CallBudget(config.max_calls or environment.max_calls)
        self.max_steps = config.max_steps or environment.max_steps
        self.episode = Episode(
            environment.name,
            task,
            config.method,
            rollout_id=rollout_id,
            configuration=asdict(config),
        )
        self.actor_window = config.history_frames
        self.planner_window = config.history_frames

    @property
    def step(self):
        return len(self.episode.actions)

    @property
    def frame(self):
        return self.episode.frames[-1]

    @property
    def prompts(self):
        return self.environment.prompt(self.config.method, self.frame)

    @property
    def observations(self):
        return [
            f.text if self.prompts.obs == "text" else f.image_b64
            for f in self.episode.frames
        ]

    def observe(self, frame):
        if not isinstance(frame, Frame):
            raise InfrastructureError("Sandbox must return Frame")
        if not frame.image_b64 and not frame.text:
            raise InfrastructureError("Sandbox returned an empty observation")
        self.episode.frames.append(frame)
        self.episode.solved = frame.solved

    def ask(self, role, messages, parser=None, **metadata):
        image_count = sum(
            item.get("type") == "image_url"
            for message in messages
            if isinstance(message["content"], list)
            for item in message["content"]
        )
        if image_count > self.config.history_frames:
            raise ValueError("Model request exceeds the visible-frame limit")
        self.budget.spend()
        turn = dict(
            turn=len(self.episode.turns),
            step=self.step,
            role=role,
            raw="",
            parsed=None,
            messages=copy.deepcopy(messages),
            error=None,
            executed=False,
            **metadata,
        )
        self.episode.turns.append(turn)
        try:
            client = self.clients["actor" if role == "actor" else "planner"]
            raw = client.complete(messages)
            if not isinstance(raw, str) or not raw.strip():
                raise ModelError("Model returned no text")
            turn["raw"] = raw
            if parser:
                try:
                    parsed = parser(raw)
                    if parsed is None:
                        raise ValueError("Invalid response")
                    turn["parsed"] = parsed
                except (ValueError, TypeError, KeyError) as exc:
                    raise ProtocolError(str(exc)) from exc
        except Exception as exc:
            turn["error"] = type(exc).__name__
            raise
        return turn

    def act(self, action, turn):
        if not self.environment.validate_action(action):
            raise ProtocolError("Invalid environment action")
        if self.step >= self.max_steps:
            raise RuntimeError("Action budget exceeded")
        frame = self.sandbox.step(action)
        self.episode.actions.append(action)
        turn["executed"] = True
        self.observe(frame)

    def done(self):
        if self.frame.solved:
            self.episode.end = "solved"
            return True
        if self.frame.terminated or self.frame.truncated:
            self.episode.end = "terminated" if self.frame.terminated else "truncated"
            return True
        if self.step >= self.max_steps:
            self.episode.end = "action_limit"
            return True
        return False


def run(
    environment: str,
    task: Task,
    planner: ModelClient,
    sandbox: Sandbox,
    config: RunConfig | None = None,
    actor: ModelClient | None = None,
    rollout_id=0,
) -> Episode:
    config = config or RunConfig()
    if config.assertion_visibility == "auto":
        config = replace(
            config,
            assertion_visibility="independent",
        )
    if config.system_intervention is None:
        config = replace(
            config, system_intervention=config.method in ("conpact_i", "conpact_s"),
        )
    spec = get(environment, task)
    if config.method not in spec.prompts:
        raise ValueError(f"{config.method} is unavailable for {environment}")
    session = Session(
        spec, task, config, planner, actor or planner, sandbox, rollout_id
    )
    try:
        session.observe(sandbox.reset(task))
        if config.method.startswith("conpact_"):
            from .methods.conpact_i import rollout

            rollout(session, intervention=config.system_intervention)
        else:
            from .baselines.registry import rollout

            rollout(session)
    except CallBudgetExhausted:
        session.episode.end = "model_call_limit"
    except (ProtocolError, ModelError):
        session.episode.end = "model_failure"
    except Exception as exc:
        session.episode.end = "infrastructure_error"
        session.episode.error = type(exc).__name__
    finally:
        session.episode.model_calls = session.budget.used
        if session.episode.end != "infrastructure_error" and hasattr(
            sandbox, "evaluate"
        ):
            try:
                outcome = sandbox.evaluate()
                session.episode.metrics.update(outcome)
                if "solved" in outcome:
                    session.episode.solved = bool(outcome["solved"])
            except Exception as exc:
                session.episode.end = "infrastructure_error"
                session.episode.error = type(exc).__name__
        try:
            sandbox.close()
        except Exception as exc:
            session.episode.end = "infrastructure_error"
            session.episode.error = type(exc).__name__
    return session.episode
