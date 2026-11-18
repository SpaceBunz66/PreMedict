"""Lịch VN (plan §5): âm lịch, ngày Tết, thứ Bảy làm bù là ngày làm việc, kỳ nghỉ, lễ theo tỉnh."""
from datetime import date

import holidays
import pandas as pd
import pytest
from holidays.calendars.chinese import _ChineseLunisolar as L

from forecast_v2 import calendar_vn as cal

TET = ['2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10', '2025-01-29', '2026-02-17', '2027-02-06']


@pytest.mark.parametrize('table,vn_exceptions,day_month', [
    (L.LUNAR_NEW_YEAR_DATES, L.VIETNAMESE_CALENDAR_LUNAR_NEW_YEAR_DATES, (1, 1)),
    (L.HUNG_KINGS_DATES, L.VIETNAMESE_CALENDAR_HUNG_KINGS_DATES, (10, 3)),
])
def test_lunar_matches_holidays_tables(table, vn_exceptions, day_month):
    """Đối chiếu 1901–2099; bảng VIETNAMESE_CALENDAR_* là các năm âm lịch VN (UTC+7) lệch với Trung Quốc."""
    for y, (m, d) in (table | vn_exceptions).items():
        assert cal.solar_to_lunar(date(y, m, d)) == (*day_month, y, 0), y


def test_tet_and_leap_months():
    assert cal.tet_days(range(2021, 2028)).strftime('%Y-%m-%d').tolist() == TET
    assert cal.solar_to_lunar(date(2023, 3, 22)) == (1, 2, 2023, 1)  # tháng 2 nhuận Quý Mão
    assert cal.solar_to_lunar(date(2025, 7, 25)) == (1, 6, 2025, 1)  # tháng 6 nhuận Ất Tỵ


def test_csv_covers_holidays_library():
    lib = pd.DatetimeIndex(sorted(holidays.VN(years=range(2021, 2028))))
    assert lib.isin(cal.days_off(range(2021, 2028))).all()


def test_features():
    f = cal.features(pd.date_range('2025-04-20', '2026-03-10'))
    assert f.loc['2025-04-26', 'is_lam_bu'] and not f.loc['2025-04-26', 'is_day_off']  # thứ Bảy làm bù
    assert f.loc['2025-05-02', 'is_day_off'] and f.loc['2025-05-02', 'holiday_type'] == 'nghi_bu'
    assert f.loc['2025-05-01', 'break_len'] == 5 and f.loc['2025-04-30', 'break_pos'] == 1  # 30/4–4/5
    assert f.loc['2026-02-17', 'break_len'] == 9 and f.loc['2026-02-14', 'break_pos'] == 1  # Tết 14–22/02 gồm cuối tuần
    assert f.loc['2026-02-10', 'days_to_break'] == 4 and f.loc['2026-02-23', 'days_since_break'] == 1
    assert f.loc['2026-02-17', 'tet_offset'] == 0 and f.loc['2026-02-10', 'tet_offset'] == -7 and f.loc['2026-02-10', 'tet_pre']
    assert f.loc['2026-03-03', ['lunar_day', 'lunar_month', 'is_ram']].tolist() == [15, 1, True]
    assert f.loc['2025-07-15', 'school_break'] and not f.loc['2025-09-15', 'school_break']
    assert f.loc['2025-06-01', 'tet_offset'] == 31  # xa Tết


def test_regional_and_coverage():
    ds = pd.Series(pd.to_datetime(['2025-02-10', '2025-02-10', '2025-04-15']))
    assert cal.regional(ds, pd.Series(['24', '01', '96'])).tolist() == [True, False, True]  # Hội Lim, Chol Chnam Thmay
    with pytest.raises(ValueError, match='2028'):
        cal.features(pd.date_range('2027-12-30', '2028-01-02'))
