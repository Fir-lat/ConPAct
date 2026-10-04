from __future__ import annotations


class CallBudgetExhausted(Exception):
    pass


class CallBudget:
    def __init__(self, limit: int):
        if limit < 0:
            raise ValueError(f"call budget must be nonnegative, got {limit}")

        self.limit = limit
        self.used = 0

    @property
    def remaining(self) -> int:

        if not self.limit:
            return -1
        return self.limit - self.used

    def spend(self, n: int = 1) -> None:

        if n < 1:
            raise ValueError(f"a call batch must be at least 1, got {n}")
        if self.limit and self.used + n > self.limit:
            raise CallBudgetExhausted(
                f"{self.used} of {self.limit} model calls used; "
                f"the next stage needs {n} more"
            )
        self.used += n
