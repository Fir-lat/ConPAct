from dataclasses import dataclass
from .base import PromptSet, fill


@dataclass(frozen=True)
class HiPlanPromptSet(PromptSet):
    hint_system: str = ""
    planner_instruction: str = ""
    hint_instruction: str = ""
    hint_instruction_with_history: str = ""


PLANNER = """You are the PLANNER of an agent. You do not take actions yourself. Use the task goal and initial observation to understand the task.

{context}

# Your Task
Create an ordered milestone guide for achieving the task goal. Each milestone should describe a concrete intermediate outcome. Use as few milestones as needed to describe a coherent path. A milestone is a high-level objective, not a sequence of individual actions. Ground the milestones in the task and available observation. State what must be true for each milestone to count as complete. Later milestones should build on earlier outcomes.
The guide is created once from this initial observation and is never rewritten during the episode.
{objective}

# Output Format
Output exactly one <plan>...</plan> block containing a numbered milestone list. You may OPTIONALLY include a <thought>...</thought> block before it. Do NOT output an action or text outside these blocks.

Example:
<thought>...</thought>
<plan>
1. ... Completion condition: ...
2. ... Completion condition: ...
</plan>
"""

HINT = """You provide step-wise guidance to the ACTOR of an agent. Use the current observation and supplied history to understand what has actually happened.

{context}

# Milestone Action Guide
{global_plan}

# Current Milestone
Milestone {milestone_index}: {subgoal}

# Your Task
Assess the supplied milestone using the current observation, BEFORE the next action. Mark it ACHIEVED only if its completion condition is already satisfied by the current observation. An intended action or previous hint is not evidence of success. Otherwise mark it CONTINUE and explain what remains to be done. Do not select a later milestone or rewrite the guide yourself.
{objective}

Provide a short hint with these named fields:
- Current State: Describe the observed situation relevant to the current milestone.
- Current Milestone: Restate the supplied milestone and its intended outcome.
- Milestone Gap: Explain the observed conditions still missing for completion. If already complete, say so. If progress is blocked, describe the obstacle instead of claiming completion.
- Action Correction: Include only when the most recent action was ineffective, counterproductive, or deviated from the milestone and its result is observable. Omit on the first step or when no correction is needed.
Ground the hint in available observations. Distinguish uncertainty from observed facts. Leave the exact next action to the actor.

# Output Format
Output exactly two blocks in this order, with no other text:
<status>CONTINUE</status> or <status>ACHIEVED</status>, followed by
<hint>
Current State: ...
Current Milestone: ...
Milestone Gap: ...
</hint>
Do NOT output an <action> block. Action Correction is optional as specified above.
"""

ACTOR = """You are the ACTOR of an agent. Execute the current milestone using the guide and step-wise hint. Observe the current environment and execute one action at a time.

{context}

# Milestone Action Guide
{global_plan}

# Current Milestone
Milestone {milestone_index}: {subgoal}

# Step-wise Hint
{step_hint}

# Your Task
Take the single next action that advances the current milestone using the current observation, available history, and hint. Do not choose the next milestone or rewrite the guide.
{objective}

# Output Format
Output exactly one <action>...</action> block containing one legal action from the action space above. You may OPTIONALLY include a <thought>...</thought> block before it. Do NOT output a milestone status, repeat the hint, or write text outside these blocks.
"""


def build_prompts(
    *, rules: str, action_space: str, objective: str, observation: str = ""
) -> HiPlanPromptSet:
    context = rules.strip() + "\n\n# Action Space\n" + action_space.strip()
    prefix = observation + "\n" if observation else ""
    return HiPlanPromptSet(
        kind="hiplan",
        planner_system=fill(PLANNER, context=context, objective=objective),
        hint_system=fill(HINT, context=context, objective=objective),
        actor_system=fill(ACTOR, context=context, objective=objective),
        planner_instruction=prefix + "Game started. Please create the milestone guide.",
        hint_instruction=prefix
        + "Assess the current milestone and provide the step-wise hint.",
        hint_instruction_with_history=prefix
        + "Assess the current milestone and provide the step-wise hint.\nPrevious actions and observations:\n",
        instruction=prefix + "Please make a move.",
        instruction_with_history=prefix
        + "Please make a move.\nPrevious actions and observations:\n",
        current_frame_block="Current frame:\n",
    )
