"""ProphetProd lấy đúng dữ liệu như production (test.py:428-485, read_pg.py:305-352); Prophet thật chỉ chạy ở backtest."""
import numpy as np
import pandas as pd

from forecast_v2.models import prophet_prod

DAYS = pd.date_range('2025-06-20', '2025-07-20')
CUT = pd.Timestamp('2025-07-15')


def test_prod_inputs_match_production_query():
    raw = pd.DataFrame({'ma_tinh': np.where(DAYS < '2025-07-01', '33', '34'), 'csyt_id': 'x', 'ds': DAYS,
                        'tong': 10.0, 'cap_cuu': 0.0})
    raw = raw[raw.ds != '2025-07-05']  # ngày không có bản ghi
    static = pd.DataFrame({'unique_id': ['x|tong', 'x|cap_cuu', 'y|tong'], 'csyt_id': ['x', 'x', 'y'],
                           'field': ['tong', 'cap_cuu', 'tong'], 'ma_tinh': pd.Categorical(['34', '34', '01'])})
    tong, cap, missing = prophet_prod.prod_inputs(raw, static, CUT)
    assert tong.ds.min() == pd.Timestamp('2025-07-01') and tong.ds.max() == CUT  # chỉ mã tỉnh mới nhất, tới cutoff
    assert pd.Timestamp('2025-07-05') not in set(tong.ds) and len(tong) == 14  # ngày thiếu không được điền
    assert list(tong.columns) == ['ds', 'y'] and (cap.y == 0).all() and missing.empty


def test_fit_predict_skips_like_production():
    dates = pd.date_range(CUT, periods=3)
    zeros = pd.DataFrame({'ds': DAYS, 'y': 0.0})
    assert (prophet_prod.fit_predict(zeros, dates) == 0).all()  # toàn 0: không train, dự báo 0
    assert (prophet_prod.fit_predict(zeros.head(1).assign(y=5.0), dates) == 0).all()  # < 2 dòng
