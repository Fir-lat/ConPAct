from ...base import PromptSet, fill
from ..actions import action_space, valid_actions
from .common import environment_rules

PLANNER_TEMPLATE = """\
You are the PLANNER of a game-playing agent. You do not take actions yourself.
Observe the current game screen to identify the character and key objects.

{environment_rules}

# Your Task
Look at the current screen and decide the next SHORT-TERM subgoal for the actor to execute.
A subgoal is a small high-level objective that can typically be achieved in a few actions.
A subgoal is NOT the full task and NOT a single low-level action.
If you are shown a previous subgoal, it has just ended, and you are told how it ended; pick what comes next.
You are also shown the subgoals adopted so far, each stamped with the step it was adopted. A large gap before the current step means that subgoal ran for a long time. The recent ones come with the frame the screen was in when you adopted them: compare those frames against the current one to see what actually changed while each subgoal was running. If a subgoal ran and the screen looks the same afterwards, restating it will not work; pick a different approach.

# Output Format
Output exactly one <subgoal>...</subgoal> block. You may OPTIONALLY include a <thought> tag before it.
Do NOT output an action.

Example:
<thought>...</thought>
<subgoal>...</subgoal>
"""

ACTOR_TEMPLATE = """\
You are the ACTOR of a game-playing agent. You execute a subgoal handed to you by the planner; you do not plan.
Observe the current game screen to identify your character and key objects. Execute the actions frame-by-frame.

{environment_rules}

# Your Subgoal
{subgoal}

# Action Space
{action_space}

# Your Task
Take the single next action that advances your subgoal, and report the status of the subgoal after that action.
You do not choose the next subgoal, the planner does. Reporting anything other than CONTINUE hands control back to the planner, which is the right move whenever you are no longer able to make progress on this subgoal. Do not keep executing a subgoal you cannot advance.

# Output Format
You may OPTIONALLY start with a <thought>...</thought> block when the situation changes or requires reasoning. Omit the thought when continuing an obvious or repetitive action.
After the optional thought, you MUST output EXACTLY TWO blocks, in the order shown below:
1) An <action>...</action> block (MUST contain exactly one of: {action_names});
2) A <status>...</status> block, containing exactly one of the following five words:
   - CONTINUE: the subgoal is still in progress and your actions are advancing it;
   - ACHIEVED: the subgoal is accomplished on the current screen;
   - IMPOSSIBLE: the subgoal cannot be accomplished as stated, for example the object it names is behind a wall or a hazard you cannot get past from here;
   - STUCK: you have been acting on this subgoal but the screen is not changing, or you keep repeating the same moves;
   - CHANGED: the screen is no longer what the subgoal assumed, an object or target is not where the subgoal implies, or another object moved and a different one now matters more.
Report CONTINUE only when you are genuinely advancing. If you cannot tell which of the other four applies, pick the closest one rather than defaulting to CONTINUE.

Example:
<thought>...</thought>
<action>{example_action}</action>
<status>CONTINUE</status>
"""


def build(game: str, *, task_instruction: str = "") -> PromptSet:
    rules = environment_rules(game, task_instruction=task_instruction)
    actions = valid_actions(game)
    return PromptSet(
        planner_system=fill(PLANNER_TEMPLATE, environment_rules=rules),
        actor_system=fill(
            ACTOR_TEMPLATE,
            environment_rules=rules,
            action_space=action_space(game),
            action_names=", ".join(actions),
            example_action="down" if "down" in actions else "right",
        ),
        kind="plan_act",
        instruction="Game started. Please make a move.",
        instruction_with_history="Game started. Please make a move.\nprevious actions and observations:\n",
        planner_prev_block="The previous subgoal was: {subgoal}\nThe actor ended it with status {status}. Decide the next subgoal.\n",
        history_subgoal_block="Subgoals adopted so far, with the step each one was adopted (oldest first):\n{items}\n",
        history_subgoal_item="- step {step}: {subgoal}",
        history_frame_block="Your recent subgoals, each followed by the screen as it was when you adopted it (oldest first):\n",
        history_frame_item="- step {step}: {subgoal}\n",
        current_frame_block="Current frame:\n",
        subgoal_fallback="(no explicit subgoal; act reactively)",
    )
