"""Lịch Việt Nam cho P1.

shortcut: tạm dùng holidays.VN, thiếu một phần ngày nghỉ bù và thứ Bảy làm bù theo thông báo hằng năm;
P4 thay bằng calendar/vn_holidays.csv.
"""
import holidays
import pandas as pd


def _vn(years):
    return holidays.VN(years=list(years), language='en_US')


def days_off(years):
    """Ngày nghỉ chính thức toàn quốc."""
    return pd.DatetimeIndex(sorted(_vn(years)))


def tet_days(years):
    """Mùng 1 Tết của từng năm."""
    return pd.DatetimeIndex(sorted(d for d, name in _vn(years).items() if name == 'Lunar New Year'))


def in_tet_window(ds, window):
    """True nếu ngày nằm trong [mùng 1 + window[0], mùng 1 + window[1]] (đơn vị: ngày)."""
    out = pd.Series(False, index=ds.index)
    for t in tet_days(range(ds.dt.year.min(), ds.dt.year.max() + 2)):
        out |= ds.between(t + pd.Timedelta(days=window[0]), t + pd.Timedelta(days=window[1]))
    return out
