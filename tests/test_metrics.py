"""Baseline naive và metric (plan §5): WAPE của seasonal-naive khớp phép tính tay."""
import numpy as np
import pandas as pd

from forecast_v2 import metrics
from forecast_v2.models import baselines

CUT = pd.Timestamp('2025-03-30')  # Chủ nhật
DAYS = pd.date_range(CUT - pd.Timedelta(days=27), CUT + pd.Timedelta(days=7))


def fold_df():
    rng = np.random.default_rng(1)
    df = pd.concat([pd.DataFrame({'unique_id': u, 'ds': DAYS, 'y': rng.integers(10, 50, len(DAYS)).astype(float),
                                  'available_mask': 1.0}) for u in ('a', 'b')], ignore_index=True)
    df.loc[(df.unique_id == 'b') & (df.ds == CUT - pd.Timedelta(days=7)), 'available_mask'] = 0  # mean4 phải bỏ ngày này
    return df


def test_naive_forecasts():
    df = fold_df()
    s = df.set_index(['unique_id', 'ds']).y
    fc = baselines.snaive7(df, None, CUT, {}).set_index(['unique_id', 'ds'])
    m4 = baselines.mean4(df, None, CUT, {}).set_index(['unique_id', 'ds'])
    d = CUT + pd.Timedelta(days=1)  # thứ Hai
    assert fc.q50[('a', d)] == s[('a', d - pd.Timedelta(days=7))]
    assert np.isclose(m4.q50[('a', d)], np.mean([s[('a', d - pd.Timedelta(days=7 * k))] for k in (1, 2, 3, 4)]))
    sun = [CUT - pd.Timedelta(days=7 * k) for k in (0, 2, 3)]  # Chủ nhật CUT − 7 bị mask
    assert np.isclose(m4.q50[('b', CUT + pd.Timedelta(days=7))], np.mean([s[('b', x)] for x in sun]))
    assert len(fc) == 14 and fc.q10.isna().all()

    df2 = df.copy()
    tue = df2.ds.dt.dayofweek.eq(1) & df2.ds.between(CUT - pd.Timedelta(days=20), CUT)  # 3 thứ Ba gần nhất bị mask
    df2.loc[tue, 'available_mask'] = 0
    t = CUT + pd.Timedelta(days=2)
    assert np.isclose(baselines.mean4(df2, None, CUT, {}).set_index(['unique_id', 'ds']).q50[('a', t)],
                      s[('a', t - pd.Timedelta(days=28))])  # chỉ còn thứ Ba thứ 4 về trước


def test_wape_mase_coverage_by_hand():
    df = fold_df()
    fc = baselines.snaive7(df, None, CUT, {})
    e = df[df.ds > CUT].merge(fc, on=['unique_id', 'ds'])
    scale = metrics.mase_scale(df, CUT)
    e = e.assign(model='snaive7', set='test', scale=e.unique_id.map(scale))

    past = df[df.ds <= CUT].set_index(['unique_id', 'ds']).y.unstack(0)
    ok = df[df.ds <= CUT].set_index(['unique_id', 'ds']).available_mask.unstack(0)
    by_hand = (past - past.shift(7)).abs().where((ok > 0) & (ok.shift(7) > 0)).mean()
    assert np.allclose(scale.sort_index(), by_hand.sort_index())

    r = metrics.score(e).iloc[0]
    assert np.isclose(r.wape, (e.q50 - e.y).abs().sum() / e.y.sum())
    assert np.isclose(r.bias, (e.q50 - e.y).sum() / e.y.sum())
    assert np.isclose(r.mase, ((e.q50 - e.y).abs() / e.scale).mean())
    assert np.isnan(r.cover80) and np.isnan(r.pinball)  # naive không có khoảng tin cậy

    q = e.assign(q10=e.y - 1, q90=e.y + 1)
    q.loc[q.index[:3], 'q90'] = q.y[:3] - 0.5  # 3/14 điểm nằm ngoài khoảng
    assert np.isclose(metrics.score(q).iloc[0].cover80, 11 / 14)


def test_interval_calibration_hits_target():
    rng = np.random.default_rng(0)
    n = 6000
    q50 = rng.uniform(10, 1000, n)
    y = q50 * np.exp(rng.normal(0, 0.3, n))  # khoảng model quá hẹp: chỉ ±10%
    e = pd.DataFrame({'model': 'm', 'h': rng.integers(1, 31, n), 'y': y, 'q50': q50, 'q10': q50 * 0.9, 'q90': q50 * 1.1})
    e.loc[:9, ['q10', 'q90']] = e.loc[:9, ['q50', 'q50']].to_numpy()  # độ rộng 0 không làm hỏng phép tính
    f = metrics.interval_factors(e)
    assert list(f.index.get_level_values('h')) == metrics.H_LABELS and (f > 1).all().all()
    c = metrics.calibrate(e, f)
    assert (c.q50 == e.q50).all() and (c.q10 <= c.q50).all() and (c.q90 >= c.q50).all()
    for _, d in c.groupby(pd.cut(c.h, metrics.H_BINS), observed=True):  # đúng 10% mỗi đầu trên chính dữ liệu fit
        assert abs((d.y < d.q10).mean() - 0.1) < 0.01 and abs((d.y > d.q90).mean() - 0.1) < 0.01
    other = e.assign(model='x')
    assert metrics.calibrate(other, f)[['q10', 'q90']].equals(other[['q10', 'q90']])  # model không có hệ số: giữ nguyên
