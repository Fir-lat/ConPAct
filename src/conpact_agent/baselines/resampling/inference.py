from ..plan_act.inference import rollout as plan_act


def rollout(session):
    return plan_act(session, asserted=True, resampling=True)
