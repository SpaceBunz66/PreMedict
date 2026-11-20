"""LightGBM direct: mẫu train chỉ nhìn tới cutoff, tra cứu đúng ngày, dự báo không đổi khi đổi dữ liệu sau cutoff."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
from forecast_v2.models import lgbm  # noqa: E402
from test_no_leakage import CFG, CUT, build  # noqa: E402  (dùng lại dữ liệu tổng hợp của test leakage)

FAST = CFG | {'models': {'lgbm': CFG['models']['lgbm'] | {
    'rounds': 60, 'interval_rounds': 30, 'params': CFG['models']['lgbm']['params'] | {'deterministic': True, 'num_threads': 1}}}}
H = CFG['features']['horizon']


def test_design_lookups():
    _, df, st, _ = build(False)
    X, y, lv, train, pred, keys = lgbm.design(df, st, CUT, H, 2)
    assert (keys.ds[train] <= CUT).all() and np.isfinite(y[train]).all()
    assert pred.sum() == st.unique_id.nunique() * H and (keys.ds[pred] > CUT).all()

    s = df.set_index(['unique_id', 'ds'])
    row = np.flatnonzero(pred & (keys.unique_id == 'a|tong') & (keys.ds == CUT + pd.Timedelta(days=9)))[0]
    level = s.y.where(s.available_mask > 0)['a|tong'][CUT - pd.Timedelta(days=27):CUT].mean()  # bỏ SPIKE (mask 0)
    assert np.isclose(lv[row], level)
    assert np.isclose(X.same_weekday[row] * level, s.y[('a|tong', CUT + pd.Timedelta(days=9 - 14))])  # h = 9: lùi 14 ngày
    assert np.isclose(X.last_year[row] * level, s.y[('a|tong', CUT + pd.Timedelta(days=9 - 364))])
    assert np.isclose(X.lag0[row] * level, s.y[('a|tong', CUT)]) and X.h[row] == 9


def test_forecast_ignores_data_after_cutoff():
    out = []
    for perturb in (False, True):
        _, df, st, _ = build(perturb)
        out.append(lgbm.forecast(df, st, CUT, FAST))
    a, b = out
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 4 * H and (a.q10 <= a.q50).all() and (a.q50 <= a.q90).all() and (a.q10 >= 0).all()

    _, df, _, _ = build(False)
    e = a.merge(df[df.available_mask > 0], on=['unique_id', 'ds'])
    assert (e.q50 - e.y).abs().sum() / e.y.sum() < 0.1  # chuỗi tuần đều 40/20/5: phải học được


def test_saved_models_reproduce_forecast(tmp_path):
    import lightgbm as lgb
    _, df, st, _ = build(False)
    out = lgbm.forecast(df, st, CUT, FAST, save_to=tmp_path)
    X, _, lv, _, pred, _ = lgbm.design(df, st, CUT, H, FAST['models']['lgbm']['origin_step'])
    q = np.column_stack([lgb.Booster(model_file=tmp_path / f'lgbm_{k}.txt').predict(X[pred]) for k in ('q10', 'q50', 'q90')])
    q = np.sort(np.clip(q, 0, None), axis=1) * lv[pred, None]
    np.testing.assert_allclose(out[['q10', 'q50', 'q90']].to_numpy(), q, rtol=1e-6)



def test_drop_feature_groups():
    _, df, st, _ = build(False)
    full = lgbm.design(df, st, CUT, H, 2)[0]
    X, _, lv, _, pred, keys = lgbm.design(df, st, CUT, H, 2, drop=('holiday', 'weather', 'clim'))
    gone = [k for k in full if k in lgbm.HOLIDAY or k.startswith(('w_', 'clim_'))]
    assert gone and set(X.columns) == set(full.columns) - set(gone)
    d = CUT + pd.Timedelta(days=9)
    row = np.flatnonzero(pred & (keys.unique_id == 'a|tong') & (keys.ds == d))[0]
    assert np.isclose(X.last_year[row] * lv[row], df.set_index(['unique_id', 'ds']).y[('a|tong', d - pd.Timedelta(days=364))])
    with pytest.raises(ValueError):
        lgbm.design(df, st, CUT, H, 2, drop=('nope',))
