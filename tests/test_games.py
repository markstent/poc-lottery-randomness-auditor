from datetime import date

import pytest

from poc_lottery_randomness_auditor.games import LOTTO


def test_era_boundaries():
    assert LOTTO.era_for_date(date(1994, 11, 19)).pool_size == 49
    assert LOTTO.era_for_date(date(2015, 10, 7)).pool_size == 49
    assert LOTTO.era_for_date(date(2015, 10, 10)).pool_size == 59
    assert LOTTO.era_for_date(date(2026, 6, 6)).rounds_per_draw == 1
    era3 = LOTTO.era_for_date(date(2026, 6, 10))
    assert era3.rounds_per_draw == 2 and era3.end is None


def test_dates_outside_eras_raise():
    with pytest.raises(ValueError):
        LOTTO.era_for_date(date(1994, 11, 18))
    with pytest.raises(ValueError):
        LOTTO.era_for_date(date(2015, 10, 8))  # gap day between eras
