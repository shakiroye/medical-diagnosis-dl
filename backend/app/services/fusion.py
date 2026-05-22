from __future__ import annotations


class FusionService:
    def combine(self, scores: list[float]) -> float:
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
