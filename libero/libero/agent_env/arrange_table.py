"""Evaluator-private contracts for the Arrange Table task variants."""

from __future__ import annotations

from typing import Any, Mapping


ARRANGE_TABLE_SUITE = "libero_arrange_table"
ARRANGE_TABLE_VISUAL_GOAL_TASK_ID = 0
ARRANGE_TABLE_TEXT_GOAL_TASK_ID = 1


class ArrangeTableTextGoalEvaluator:
    """Accept either one-to-one assignment of the two cups to the plates.

    LIBERO's ordinary BDDL runtime evaluates a flat conjunction of predicates.
    The textual goal intentionally leaves the cup-to-plate identity unspecified,
    so its authoritative checker must accept either bijection while retaining
    the existing global ``On`` and ``In`` predicate semantics.
    """

    _BUTTER_IN_BASKET = ("in", "butter_1", "basket_1_contain_region")
    _CUP_ASSIGNMENTS = (
        (
            ("on", "porcelain_mug_1", "plate_1"),
            ("on", "white_yellow_mug_1", "plate_2"),
        ),
        (
            ("on", "porcelain_mug_1", "plate_2"),
            ("on", "white_yellow_mug_1", "plate_1"),
        ),
    )

    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self) -> None:
        """The checker is final-state-only and has no episode history."""

    def observe(self, _raw_observation: Mapping[str, Any]) -> None:
        """No process state is needed for this final-state goal."""

    def result(self) -> dict[str, Any]:
        domain = self.env.env
        butter_in_basket = self._evaluate(domain, self._BUTTER_IN_BASKET)
        assignment_results = [
            all(self._evaluate(domain, predicate) for predicate in assignment)
            for assignment in self._CUP_ASSIGNMENTS
        ]
        return {
            "schema_version": "libero.arrange_table_private_evaluation.v1",
            "success": bool(butter_in_basket and any(assignment_results)),
            "butter_in_basket": butter_in_basket,
            "cup_assignment_results": assignment_results,
            "checker_semantics": "one_cup_per_plate_permutation_invariant",
        }

    @staticmethod
    def _evaluate(domain: Any, predicate: tuple[str, str, str]) -> bool:
        return bool(domain._eval_predicate(list(predicate)))


def arrange_table_private_evaluator(
    env: Any,
    *,
    suite: str,
    task_id: int,
) -> ArrangeTableTextGoalEvaluator | None:
    """Select the private checker only for the textual goal variant."""

    if suite == ARRANGE_TABLE_SUITE and task_id == ARRANGE_TABLE_TEXT_GOAL_TASK_ID:
        return ArrangeTableTextGoalEvaluator(env)
    return None


__all__ = [
    "ARRANGE_TABLE_SUITE",
    "ARRANGE_TABLE_TEXT_GOAL_TASK_ID",
    "ARRANGE_TABLE_VISUAL_GOAL_TASK_ID",
    "ArrangeTableTextGoalEvaluator",
    "arrange_table_private_evaluator",
]
