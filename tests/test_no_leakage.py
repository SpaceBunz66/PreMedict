"""Chống leakage (plan §5): đổi y và thời tiết sau cutoff thì dữ liệu train và covariate của fold không đổi."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecast_v2 import features

CFG = yaml.safe_load((Path(__file__).parents[1] / 'configs' / 'base.yaml').read_text(encoding='utf-8'))
DELAY = CFG['features']['weather_obs_delay_days']
DAYS = pd.date_range('2024-06-01', '2026-01-31')
CUT = pd.Timestamp('2025-12-15')
SPIKE = CUT - pd.Timedelta(days=3)  # thứ Sáu trước cutoff: gấp 4 lần thường ngày
IDS = ['a', 'b']


def inputs(perturb):
    """perturb: y sau cutoff gấp 4 và thời tiết từ cutoff − delay + 1 bị thay. Khi đó, median trượt đặt ở giữa của P1
    thấy SPIKE ngang mức các tuần sau nên thôi gắn cờ ngoại lai: mask tính trên toàn bộ dữ liệu bị đổi."""
    wd = DAYS.dayofweek.to_numpy()
    y = np.select([wd < 5, wd == 5], [40.0, 20.0], 5.0)
    y[DAYS == SPIKE] *= 4
    if perturb:
        y[DAYS > CUT] *= 4
    daily = pd.concat([pd.DataFrame({'csyt_id': c, 'ds': DAYS, **{f: y for f in CFG['fields']['direct']}})
                       for c in IDS], ignore_index=True)
    report = pd.DataFrame({f'use_{f}': f in ('tong', 'luot_kham') for f in CFG['fields']['direct']}, index=IDS)
    static = pd.DataFrame({'ma_tinh': '24', 'latitude': 21.0, 'longitude': 106.0, 'tuyen': '3', 'hang': '3',
                           'radius_km': 6, 'strict': True}, index=pd.Index(IDS, name='csyt_id'))
    rng = np.random.default_rng(0)
    weather = pd.concat([pd.DataFrame({'ds': DAYS, 'csyt_id': c, 'radius_km': 6, 't_mean': rng.normal(25, 3, len(DAYS)),
                                       'rain_mm': rng.gamma(1, 5, len(DAYS)), 'doy': DAYS.dayofyear})
                         for c in IDS], ignore_index=True)
    if perturb:
        late = weather.ds > CUT - pd.Timedelta(days=DELAY)
        weather.loc[late, ['t_mean', 'rain_mm']] = -99.0
    clim = pd.concat([pd.DataFrame({'csyt_id': c, 'radius_km': 6, 'doy': range(1, 367), 'clim_t_mean': 25.0,
                                    'clim_rh_mean': 80.0, 'clim_rain_mm': 5.0}) for c in IDS], ignore_index=True)
    return daily, report, static, weather, clim


def build(perturb):
    daily, *rest = inputs(perturb)
    long, static, last_day = features.table(daily, *rest, CFG)
    return long, *features.fold(long, static, daily, CUT, CFG), last_day


def test_fold_ignores_data_after_cutoff():
    full_a, a, sa, last_a = build(False)
    full_b, b, sb, last_b = build(True)
    assert last_a == last_b == DAYS[-1]
    at = lambda df, d: df[(df.unique_id == 'a|tong') & (df.ds == d)].iloc[0]

    # Mask tính trên toàn bộ dữ liệu có nhìn thấy tương lai: test này đủ nhạy để bắt leakage.
    assert at(full_a, SPIKE).available_mask == 0 and at(full_b, SPIKE).available_mask == 1
    assert at(a, SPIKE).available_mask == 0  # fold chỉ dùng dữ liệu tới cutoff

    past = lambda df: df[df.ds <= CUT]
    pd.testing.assert_frame_equal(past(a), past(b))
    future = lambda df: df[df.ds > CUT].drop(columns=['y', 'available_mask'])  # y và mask tương lai chỉ để đánh giá
    pd.testing.assert_frame_equal(future(a), future(b))
    pd.testing.assert_frame_equal(sa, sb)
    assert a.ds.max() == CUT + pd.Timedelta(days=CFG['features']['horizon'])


def test_weather_lag_and_future_rows():
    full, df, st, last_day = build(False)
    _, _, _, weather, _ = inputs(False)
    obs = weather[weather.csyt_id == 'a'].set_index('ds').t_mean
    row = lambda d: df[(df.unique_id == 'a|tong') & (df.ds == d)].iloc[0]
    assert np.isclose(row(CUT).w_t_mean, obs[CUT - pd.Timedelta(days=DELAY)])
    assert row(CUT + pd.Timedelta(days=1)).w_t_mean != row(CUT + pd.Timedelta(days=1)).w_t_mean  # NaN sau cutoff
    tail = full[full.ds > last_day]
    assert tail.y.isna().all() and (tail.available_mask == 0).all() and tail.ds.nunique() == CFG['features']['horizon']
    assert set(st.unique_id) == {'a|tong', 'a|luot_kham', 'b|tong', 'b|luot_kham'}
    assert st.weekend_open.eq(1).all() and st.log_scale.notna().all()
