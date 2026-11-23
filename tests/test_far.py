"""train-far: chuỗi thiếu dữ liệu gần gốc dùng số cùng kỳ năm trước, chuỗi đủ dữ liệu giữ dự báo model."""
import numpy as np
import pandas as pd

from forecast_v2 import backtest

CUT = pd.Timestamp('2026-06-07')
DAYS = pd.date_range(CUT - pd.Timedelta(days=400), CUT)


def test_thin_fallback():
    df = pd.concat([pd.DataFrame({'unique_id': u, 'ds': DAYS, 'y': np.arange(len(DAYS), dtype=float), 'available_mask': 1.0})
                    for u in ('ok', 'thin')], ignore_index=True)
    df.loc[(df.unique_id == 'thin') & (df.ds > CUT - pd.Timedelta(days=28 - 7)), 'available_mask'] = 0  # còn 7/28 ngày
    fut = pd.date_range(CUT + pd.Timedelta(days=169), periods=30)
    fc = pd.DataFrame({'unique_id': np.repeat(['ok', 'thin'], 30), 'ds': np.tile(fut, 2), 'q50': 1000.0})
    fc = fc.assign(q10=fc.q50 * 0.8, q90=fc.q50 * 1.3)
    out, thin = backtest.thin_fallback(fc, df, CUT)
    assert thin == ['thin']
    ok, th = out[out.unique_id == 'ok'], out[out.unique_id == 'thin']
    assert (ok.q50 == 1000).all()
    y = df[df.unique_id == 'thin'].set_index('ds').y
    np.testing.assert_allclose(th.q50, y.reindex(th.ds - pd.Timedelta(days=364)).to_numpy())
    np.testing.assert_allclose(th.q10 / th.q50, 0.8)
    np.testing.assert_allclose(th.q90 / th.q50, 1.3)
