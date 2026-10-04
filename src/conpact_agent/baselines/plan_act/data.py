from ...data.selection import ordinary_sft


def build(episodes):
    return ordinary_sft(episodes, "plan_act")
