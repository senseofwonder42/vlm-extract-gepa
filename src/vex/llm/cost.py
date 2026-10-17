"""Token usage and USD cost, per call and per run, with a budget guard.

OpenRouter returns the exact charged amount in `usage.cost`; that value is used
whenever present. Otherwise the cost is computed from the YAML price table.
Cache hits are recorded with `usd = 0` so call counts stay complete.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass

from vex.config import Price


class BudgetExceeded(RuntimeError):
    """Raised before a call once the run has spent `budget.max_usd`."""


@dataclass(frozen=True)
class CallUsage:
    """Usage of one model call.

    Attributes:
        role: Role that made the call (extractor, diagnostic, ...).
        model: Model id.
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens, reasoning included.
        reasoning_tokens: Reasoning part of `completion_tokens`.
        usd: Charged cost (0 for cache hits).
        cached: True if served from the disk cache.
    """

    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    usd: float
    cached: bool


def usage_from_response(
    response: dict, role: str, model: str, prices: dict[str, Price], cached: bool
) -> CallUsage:
    """Build a `CallUsage` from an OpenRouter chat-completion response.

    Args:
        response: Parsed JSON response.
        role: Calling role.
        model: Requested model id.
        prices: Fallback prices per model (USD per million tokens).
        cached: Whether the response came from the cache.

    Returns:
        The call usage.
    """
    usage = response.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    if cached:
        usd = 0.0
    elif usage.get("cost") is not None:
        usd = float(usage["cost"])
    elif model in prices:
        usd = (prompt * prices[model].input + completion * prices[model].output) / 1e6
    else:
        usd = 0.0
    return CallUsage(role, model, prompt, completion, reasoning, usd, cached)


class CostTracker:
    """Accumulates call usage for one run and enforces the budget."""

    def __init__(self, max_usd: float) -> None:
        """Create a tracker.

        Args:
            max_usd: Spending limit for the run.
        """
        self.max_usd = max_usd
        self.calls: list[CallUsage] = []

    @property
    def total_usd(self) -> float:
        """Total spent so far."""
        return sum(call.usd for call in self.calls)

    def record(self, usage: CallUsage) -> None:
        """Add one call.

        Args:
            usage: The call usage.
        """
        self.calls.append(usage)

    def check_budget(self) -> None:
        """Refuse new calls once the budget is spent.

        Raises:
            BudgetExceeded: If `total_usd >= max_usd`.
        """
        if self.total_usd >= self.max_usd:
            raise BudgetExceeded(f"Budget reached: ${self.total_usd:.4f} spent, limit ${self.max_usd:.2f}")

    def summary(self) -> dict:
        """Return totals overall and per role, ready to dump as `cost.json`."""
        by_role: dict[str, dict[str, float]] = defaultdict(
            lambda: dict.fromkeys(
                ("calls", "cached", "prompt_tokens", "completion_tokens", "reasoning_tokens", "usd"), 0
            )
        )
        for call in self.calls:
            row = by_role[call.role]
            row["calls"] += 1
            row["cached"] += int(call.cached)
            row["prompt_tokens"] += call.prompt_tokens
            row["completion_tokens"] += call.completion_tokens
            row["reasoning_tokens"] += call.reasoning_tokens
            row["usd"] += call.usd
        return {
            "total_usd": self.total_usd,
            "max_usd": self.max_usd,
            "calls": len(self.calls),
            "by_role": dict(by_role),
            "per_call": [asdict(call) for call in self.calls],
        }
