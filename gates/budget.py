"""Invocation budget plus worst-case forecast reservation, without shortening reviews."""


class BudgetLedger:
    def __init__(self, invocations, forecasts):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (invocations, forecasts)):
            raise ValueError("Nonnegative finite budgets required")
        self.invocations, self.forecasts = invocations, forecasts
        self.pending = {}
        self.seen = set()

    def reserve(self, decision_id, maximum):
        if decision_id in self.seen: raise ValueError("Duplicate invocation")
        if self.invocations < 1 or self.forecasts < maximum: raise ValueError("Insufficient reserved WM budget")
        self.seen.add(decision_id)
        self.pending[decision_id] = maximum
        self.invocations -= 1
        self.forecasts -= maximum

    def settle(self, decision_id, actual):
        reserved = self.pending[decision_id]
        if isinstance(actual, bool) or not isinstance(actual, int) or not 1 <= actual <= reserved:
            raise ValueError("Actual forecasts exceed fixed review protocol reservation")
        self.forecasts += reserved - actual
        del self.pending[decision_id]

    def snapshot(self):
        return {"invocations_remaining": self.invocations, "forecasts_remaining": self.forecasts,
                "unsettled_reservations": dict(self.pending)}
