"""Lịch Việt Nam (plan §P4): ngày nghỉ, Tết, âm lịch, lễ theo tỉnh, nghỉ hè; feature lịch biết trước cho mọi horizon.

Nguồn ngày lễ: calendar/vn_holidays.csv (một dòng = một khoảng ngày; scope `national` hoặc `province:<mã,...>`).
"""
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAYS_CSV = Path(__file__).resolve().parents[2] / 'calendar' / 'vn_holidays.csv'
OFF_TYPES = ['tet', 'national', 'nghi_bu']  # thứ tự ưu tiên khi một ngày có nhiều loại

# ---- Âm lịch Việt Nam, múi giờ UTC+7. Thuật toán của Hồ Ngọc Đức (amlich), giữ nguyên hằng số. ----
_TZ = 7


def _new_moon(k):
    """Ngày Julian (thực) của lần sóc thứ k kể từ 1900-01-01."""
    t = k / 1236.85
    t2, t3 = t * t, t * t * t
    dr = math.pi / 180
    jd1 = 2415020.75933 + 29.53058868 * k + 0.0001178 * t2 - 0.000000155 * t3
    jd1 += 0.00033 * math.sin((166.56 + 132.87 * t - 0.009173 * t2) * dr)
    m = 359.2242 + 29.10535608 * k - 0.0000333 * t2 - 0.00000347 * t3
    mpr = 306.0253 + 385.81691806 * k + 0.0107306 * t2 + 0.00001236 * t3
    f = 21.2964 + 390.67050646 * k - 0.0016528 * t2 - 0.00000239 * t3
    c1 = (0.1734 - 0.000393 * t) * math.sin(m * dr) + 0.0021 * math.sin(2 * dr * m)
    c1 += -0.4068 * math.sin(mpr * dr) + 0.0161 * math.sin(dr * 2 * mpr) - 0.0004 * math.sin(dr * 3 * mpr)
    c1 += 0.0104 * math.sin(dr * 2 * f) - 0.0051 * math.sin(dr * (m + mpr))
    c1 += -0.0074 * math.sin(dr * (m - mpr)) + 0.0004 * math.sin(dr * (2 * f + m))
    c1 += -0.0004 * math.sin(dr * (2 * f - m)) - 0.0006 * math.sin(dr * (2 * f + mpr))
    c1 += 0.0010 * math.sin(dr * (2 * f - mpr)) + 0.0005 * math.sin(dr * (2 * mpr + m))
    deltat = (0.001 + 0.000839 * t + 0.0002261 * t2 - 0.00000845 * t3 - 0.000000081 * t * t3 if t < -11
              else -0.000278 + 0.000265 * t + 0.000262 * t2)
    return jd1 + c1 - deltat


def _new_moon_day(k):
    return math.floor(_new_moon(k) + 0.5 + _TZ / 24)


def _sun_sector(jdn):
    """Kinh độ mặt trời lúc 0h giờ địa phương của ngày jdn, chia thành 12 cung (0..11)."""
    t = (jdn - 0.5 - _TZ / 24 - 2451545.0) / 36525
    t2 = t * t
    dr = math.pi / 180
    m = 357.52910 + 35999.05030 * t - 0.0001559 * t2 - 0.00000048 * t * t2
    l0 = 280.46645 + 36000.76983 * t + 0.0003032 * t2
    dl = (1.914600 - 0.004817 * t - 0.000014 * t2) * math.sin(dr * m)
    dl += (0.019993 - 0.000101 * t) * math.sin(dr * 2 * m) + 0.000290 * math.sin(dr * 3 * m)
    lon = (l0 + dl) * dr
    lon -= 2 * math.pi * math.floor(lon / (2 * math.pi))
    return math.floor(lon / math.pi * 6)


@lru_cache(maxsize=None)
def _month11(yy):
    """Ngày bắt đầu tháng 11 âm lịch (tháng chứa Đông chí) của năm yy."""
    k = math.floor((date(yy, 12, 31).toordinal() + 1721425 - 2415021) / 29.530588853)
    nm = _new_moon_day(k)
    return _new_moon_day(k - 1) if _sun_sector(nm) >= 9 else nm


@lru_cache(maxsize=None)
def _leap_offset(a11):
    k = math.floor((a11 - 2415021.076998695) / 29.530588853 + 0.5)
    i, arc = 1, _sun_sector(_new_moon_day(k + 1))
    while True:  # tháng nhuận: tháng đầu tiên sau tháng 11 không chứa trung khí
        last, i = arc, i + 1
        arc = _sun_sector(_new_moon_day(k + i))
        if arc == last or i >= 14:
            return i - 1


def solar_to_lunar(d):
    """date dương lịch -> (ngày, tháng, năm, nhuận) âm lịch Việt Nam."""
    jdn = d.toordinal() + 1721425
    k = math.floor((jdn - 2415021.076998695) / 29.530588853)
    start = _new_moon_day(k + 1)
    if start > jdn:
        start = _new_moon_day(k)
    a11 = _month11(d.year)
    if a11 >= start:
        year, b11, a11 = d.year, a11, _month11(d.year - 1)
    else:
        year, b11 = d.year + 1, _month11(d.year + 1)
    diff = math.floor((start - a11) / 29)
    month, leap = diff + 11, 0
    if b11 - a11 > 365:
        leap_diff = _leap_offset(a11)
        if diff >= leap_diff:
            month, leap = diff + 10, int(diff == leap_diff)
    if month > 12:
        month -= 12
    if month >= 11 and diff < 4:
        year -= 1
    return jdn - start + 1, month, year, leap


def tet_days(years):
    """Mùng 1 Tết của từng năm (luôn nằm trong 21/01–20/02)."""
    return pd.DatetimeIndex([next(d for d in pd.date_range(f'{y}-01-19', f'{y}-02-22')
                                  if solar_to_lunar(d.date()) == (1, 1, y, 0)) for y in years])


def in_tet_window(ds, window):
    """True nếu ngày nằm trong [mùng 1 + window[0], mùng 1 + window[1]] (đơn vị: ngày)."""
    out = pd.Series(False, index=ds.index)
    for t in tet_days(range(ds.dt.year.min(), ds.dt.year.max() + 2)):
        out |= ds.between(t + pd.Timedelta(days=window[0]), t + pd.Timedelta(days=window[1]))
    return out


# ---- Ngày lễ từ vn_holidays.csv ----
@lru_cache(maxsize=1)
def holidays_by_day():
    """Mỗi dòng: một ngày (ds) của một mục trong vn_holidays.csv."""
    h = pd.read_csv(HOLIDAYS_CSV, parse_dates=['start', 'end'], dtype={'scope': str})
    days = [pd.date_range(s, e) for s, e in zip(h.start, h.end)]
    return h.loc[h.index.repeat([len(d) for d in days])].assign(ds=np.concatenate(days)).reset_index(drop=True)


def _check_years(years):
    h = holidays_by_day()
    covered = set(h.ds[(h.scope == 'national') & (h.type == 'national')].dt.year)
    missing = sorted(set(years) - covered)
    if missing:  # thiếu năm thì mọi ngày lễ năm đó thành ngày thường: dừng thay vì cho ra feature sai
        raise ValueError(f'calendar/vn_holidays.csv chưa có năm {missing}: bổ sung lịch nghỉ trước khi dùng')


def days_off(years):
    """Ngày nghỉ lễ chính thức toàn quốc (không tính thứ Bảy, Chủ nhật thường)."""
    _check_years(years)
    h = holidays_by_day()
    off = h[(h.scope == 'national') & h.type.isin(OFF_TYPES) & (h.is_day_off == 1)]
    return pd.DatetimeIndex(sorted(set(off.ds)))


@lru_cache(maxsize=1)
def _all_features(tet_window=14, clip=30):
    h = holidays_by_day()
    nat = h[h.scope == 'national']
    years = sorted(set(nat.ds[nat.type == 'national'].dt.year))
    dates = pd.date_range(f'{years[0]}-01-01', f'{years[-1]}-12-31', name='ds')
    f = pd.DataFrame(index=dates)
    f['dow'] = dates.dayofweek
    doy = dates.dayofyear.to_numpy() / 365.25
    for k in (1, 2):
        f[f'doy_sin{k}'], f[f'doy_cos{k}'] = np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)
    f['dom'] = dates.day
    f['days_to_month_end'] = dates.days_in_month - dates.day

    off = nat[nat.type.isin(OFF_TYPES) & (nat.is_day_off == 1)]
    f['is_holiday'] = dates.isin(off.ds)
    f['is_lam_bu'] = dates.isin(nat.ds[nat.type == 'lam_bu'])  # thứ Bảy làm bù: ngày làm việc
    f['is_day_off'] = f.is_holiday | ((f.dow >= 5) & ~f.is_lam_bu)
    first_type = off.assign(p=off.type.map(OFF_TYPES.index)).sort_values('p').drop_duplicates('ds')
    f['holiday_type'] = first_type.set_index('ds').type.reindex(dates).fillna('none')

    # Kỳ nghỉ = chuỗi ngày nghỉ liên tiếp có ít nhất một ngày lễ (vd. Tết 14–22/02/2026 tính cả cuối tuần).
    run = (f.is_day_off != f.is_day_off.shift()).cumsum()
    in_break = f.is_day_off & f.groupby(run).is_holiday.transform('any')
    f['break_len'] = np.where(in_break, f.groupby(run).dow.transform('size'), 0)
    f['break_pos'] = np.where(in_break, f.groupby(run).cumcount() + 1, 0)
    pos = pd.Series(np.where(in_break, np.arange(len(f)), np.nan))
    f['days_to_break'] = (pos.bfill() - np.arange(len(f))).fillna(clip).clip(0, clip).to_numpy()
    f['days_since_break'] = (np.arange(len(f)) - pos.ffill()).fillna(clip).clip(0, clip).to_numpy()

    tets = tet_days(range(years[0] - 1, years[-1] + 2))
    offsets = (dates.to_numpy()[:, None] - tets.to_numpy()[None, :]) // np.timedelta64(1, 'D')
    tet_offset = offsets[np.arange(len(dates)), np.abs(offsets).argmin(axis=1)]
    f['tet_offset'] = tet_offset.clip(-clip - 1, clip + 1)  # ±31: xa Tết
    f['tet_pre'] = (tet_offset >= -tet_window) & (tet_offset < 0)
    f['tet_post'] = (tet_offset > 0) & (tet_offset <= tet_window)

    lunar = np.array([solar_to_lunar(d) for d in dates.date])
    f['lunar_day'], f['lunar_month'], f['lunar_leap'] = lunar[:, 0], lunar[:, 1], lunar[:, 3]
    f['is_mung1'], f['is_ram'] = f.lunar_day == 1, f.lunar_day == 15
    f['school_break'] = dates.isin(nat.ds[nat.type == 'school'])
    return f


def features(dates):
    """Feature lịch toàn quốc theo ngày, đều biết trước cho mọi horizon. Năm chưa có trong CSV: báo lỗi."""
    dates = pd.DatetimeIndex(dates)
    _check_years(range(dates.year.min(), dates.year.max() + 1))
    return _all_features().reindex(dates)


def regional(ds, ma_tinh):
    """True nếu ngày là lễ theo tỉnh (scope province:<mã,...>, gồm cả mã tỉnh cũ và mới). ds, ma_tinh: Series cùng index."""
    h = holidays_by_day()
    r = h[h.scope.str.startswith('province:')]
    pairs = r.assign(ma_tinh=r.scope.str.removeprefix('province:').str.split(',')).explode('ma_tinh')
    known = pd.MultiIndex.from_frame(pairs[['ds', 'ma_tinh']])
    return pd.Series(pd.MultiIndex.from_arrays([ds, ma_tinh.astype(str)]).isin(known), index=ds.index)
