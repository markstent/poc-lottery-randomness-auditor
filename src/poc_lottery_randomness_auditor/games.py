"""Game definitions and Lotto era rules."""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Era:
    """A period in which a game was played under one set of rules.

    ``end`` is inclusive; ``None`` means the era is ongoing.
    """

    name: str
    start: date
    end: date | None
    pool_size: int
    picks: int
    has_bonus: bool
    rounds_per_draw: int


@dataclass(frozen=True)
class GameSpec:
    """A game defined by an ordered tuple of eras."""

    name: str
    eras: tuple[Era, ...]

    def era_for_date(self, d: date) -> Era:
        """Return the era containing ``d``.

        Raises:
            ValueError: If no era contains ``d``.
        """
        for era in self.eras:
            if era.start <= d and (era.end is None or d <= era.end):
                return era
        raise ValueError(f"no era of {self.name} contains {d}")


LOTTO = GameSpec(
    name="lotto",
    eras=(
        Era("era-49", date(1994, 11, 19), date(2015, 10, 7), 49, 6, True, 1),
        Era("era-59", date(2015, 10, 10), date(2026, 6, 6), 59, 6, True, 1),
        Era("era-59-two-round", date(2026, 6, 10), None, 59, 6, True, 2),
    ),
)
