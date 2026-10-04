import importlib

METHODS = (
    "react",
    "plan_act",
    "hiplan",
    "tape",
    "self_refine",
    "assertion",
    "resampling",
    "paa",
    "hsl",
    "webstar",
    "ecot",
)


def rollout(session):
    method = session.config.method
    if method not in METHODS:
        raise ValueError("Unknown baseline")
    module = "react" if method in ("hsl", "webstar") else method
    return importlib.import_module(f"{__package__}.{module}.inference").rollout(session)


def build(method, episodes, *, teacher=None, reviewer=None):
    if method not in ("react", "plan_act", "paa", "hsl", "webstar", "ecot"):
        raise ValueError("Unknown training baseline")
    function = importlib.import_module(f"{__package__}.{method}.data").build
    if method in ("react", "plan_act"):
        return function(episodes)
    if method in ("hsl", "webstar"):
        return function(episodes, teacher)
    return function(episodes, teacher, reviewer)
