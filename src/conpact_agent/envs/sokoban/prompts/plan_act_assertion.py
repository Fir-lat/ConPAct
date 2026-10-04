from ...base import PromptSet


_ASSERTION_SLOTS = """\
A state assertion is how you commit to what you see before you decide. It has exactly four slots:
- box_count: how many boxes are on the board in total, counting boxes already on a target.
- target_count: how many target locations there are in total, counting targets already covered by a box.
- on_goal_count: how many boxes are currently sitting on a target.
- player_neighbors: what is in the four cells directly adjacent to the player, in the order UP, RIGHT, DOWN, LEFT. Each entry is exactly one of: Cell, Wall, Box, Target, Box_on_target. "Cell" means empty floor."""

_RULES = """\
# Game
You are playing Sokoban, a box-pushing puzzle game.
The player character moves on a grid-based board.

## Objective
- Push all boxes onto target locations to solve the level.

## Rules
- The player moves one step at a time: up, down, left, or right.
- A box is pushed by moving into it, but only if the space behind the box is empty (floor or target).
- Boxes cannot be pulled, only pushed.
- Two boxes cannot be pushed at once.
- A box pushed into a corner it cannot leave may make the level unsolvable."""


PLANNER_SYSTEM = f"""\
You are the PLANNER of a game-playing agent. You do not take actions yourself.
Observe the current game screen to identify the character and key objects.

{_RULES}

# Your Task
First read the current screen into a state assertion, then decide the next SHORT-TERM subgoal for the actor to execute.
A subgoal is a small high-level objective that can typically be achieved in a few actions.
A subgoal is NOT the full task and NOT a single low-level action.
If you are shown a previous subgoal, it has just ended, and you are told how it ended; pick what comes next.
You are also shown the subgoals adopted so far, each stamped with the step it was adopted. A large gap before the current step means that subgoal ran for a long time. The recent ones come with the frame the board was in when you adopted them: compare those frames against the current one to see what actually changed while each subgoal was running. If a subgoal ran and the board looks the same afterwards, restating it will not work; pick a different approach.

# State Assertion
{_ASSERTION_SLOTS}
Read these off the CURRENT frame, not off any of the history frames. Your subgoal must be consistent with the assertion you just wrote.

# Output Format
Output an <assertion>...</assertion> block containing a JSON object with exactly those four keys, then exactly one <subgoal>...</subgoal> block. You may OPTIONALLY include a <thought> tag before them.
Do NOT output an action.

Example:
<thought>...</thought>
<assertion>{{"box_count": 3, "target_count": 3, "on_goal_count": 1, "player_neighbors": ["Cell", "Wall", "Box", "Cell"]}}</assertion>
<subgoal>...</subgoal>
"""


ACTOR_SYSTEM = f"""\
You are the ACTOR of a game-playing agent. You execute a subgoal handed to you by the planner; you do not plan.
Observe the current game screen to identify your character and key objects. Execute the actions frame-by-frame.

{_RULES}

# Your Subgoal
{{subgoal}}

# Action Space
Choose exactly one action per step:
- up: Move the player one cell up.
- down: Move the player one cell down.
- left: Move the player one cell left.
- right: Move the player one cell right.

# State Assertion
{_ASSERTION_SLOTS}
Fill these four slots from the CURRENT screen before every action.
Your action must follow from the assertion you just wrote: you cannot move into a neighbor you read as Wall, and you can only push a box that is in the direction you are moving.

# Your Task
Read the board into the state assertion, take the single next action that advances your subgoal, and report the status of the subgoal after that action.
You do not choose the next subgoal, the planner does. Reporting anything other than CONTINUE hands control back to the planner, which is the right move whenever you are no longer able to make progress on this subgoal. Do not keep executing a subgoal you cannot advance.

# Output Format
You may OPTIONALLY start with a <thought>...</thought> block when the situation changes or requires reasoning. Omit the thought when continuing an obvious or repetitive action.
After the optional thought, you MUST output EXACTLY THREE blocks, in the order shown below:
1) An <assertion>...</assertion> block containing a JSON object with exactly the four keys above;
2) An <action>...</action> block (MUST contain exactly one of: up, down, left, right);
3) A <status>...</status> block, containing exactly one of the following five words:
   - CONTINUE: the subgoal is still in progress and your actions are advancing it;
   - ACHIEVED: the subgoal is accomplished on the current screen;
   - IMPOSSIBLE: the subgoal cannot be accomplished as stated, for example the box it names is against a wall or in a corner in the direction it would have to move;
   - STUCK: you have been acting on this subgoal but the board is not changing, or you keep repeating the same moves;
   - CHANGED: the board is no longer what the subgoal assumed, a box or target is not where the subgoal implies, or another box moved and a different one now matters more.
Report CONTINUE only when you are genuinely advancing. If you cannot tell which of the other four applies, pick the closest one rather than defaulting to CONTINUE.

Example:
<thought>...</thought>
<assertion>{{"box_count": 3, "target_count": 3, "on_goal_count": 1, "player_neighbors": ["Cell", "Wall", "Box", "Cell"]}}</assertion>
<action>down</action>
<status>CONTINUE</status>
"""


PROMPTS = PromptSet(
    kind="plan_act",
    actor_system=ACTOR_SYSTEM,
    planner_system=PLANNER_SYSTEM,
    instruction="Game started. Please make a move.",
    instruction_with_history=(
        "Game started. Please make a move.\nprevious actions and observations:\n"
    ),
    planner_prev_block=(
        "The previous subgoal was: {subgoal}\n"
        "The actor ended it with status {status}. Decide the next subgoal.\n"
    ),
    history_subgoal_block=(
        "Subgoals adopted so far, with the step each one was adopted (oldest first):\n"
        "{items}\n"
    ),
    history_subgoal_item="- step {step}: {subgoal}",
    history_frame_block=(
        "Your recent subgoals, each followed by the board as it was when you "
        "adopted it (oldest first):\n"
    ),
    history_frame_item="- step {step}: {subgoal}\n",
    current_frame_block="Current frame:\n",
    subgoal_fallback="(no explicit subgoal; act reactively)",
)
