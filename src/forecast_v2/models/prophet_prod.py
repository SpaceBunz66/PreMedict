"""ProphetProd và ProphetProd-clean (plan §P6): tái hiện Prophet production, viết lại, không import code cũ.

Logic cũ (thư mục cha):
- test.py:63-66 train_prophet_model: Prophet(weekly_seasonality=True, daily_seasonality=False), không biến ngoài.
- test.py:428-485 _build_csyt_forecast: mỗi chỉ tiêu một model; chuỗi toàn 0 hoặc < 2 dòng, hoặc train lỗi: dự báo 0;
  yhat clip >= 0.
- read_pg.py:305-352 get_datamart_by_csyt: dòng theo (ma_tinh, csyt_id), GROUP BY time_id, time_id < hôm nay. Ngày không
  có bản ghi thì không có dòng (không điền), dòng 0 (kể cả 0 đệm sau ngày cuối có dữ liệu) vẫn giữ.

prod: đúng như trên tại gốc cutoff (dòng ds <= cutoff), mã tỉnh = mã mới nhất của CSYT (production chạy theo từng cặp
(ma_tinh, csyt_id), nên CSYT đổi mã tỉnh 07/2025 chỉ train trên phần dữ liệu của mã mới).
cleaned: cùng model trên chuỗi đã làm sạch (cộng dồn qua mã tỉnh, ngày mask = 0 bỏ khỏi train): tách phần lợi do sửa
dữ liệu khỏi phần lợi do đổi model.
"""
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from forecast_v2 import clean


def fit_predict(train, dates):
    """train: DataFrame [ds, y]; dates: ngày cần dự báo. Như vòng lặp chỉ tiêu của _build_csyt_forecast."""
    train = train[['ds', 'y']].dropna()
    if train.y.sum() == 0 or len(train) < 2:
        return np.zeros(len(dates))
    from prophet import Prophet
    logging.getLogger('cmdstanpy').disabled = True  # setLevel như test.py:30-31 không chặn được log INFO trong tiến trình con
    try:
        m = Prophet(weekly_seasonality=True, daily_seasonality=False)
        m.fit(train)
        return m.predict(pd.DataFrame({'ds': dates})).yhat.clip(lower=0).to_numpy()
    except Exception as e:  # như code cũ: lỗi train thì dự báo 0
        logging.warning(f'Prophet lỗi: {e}')
        return np.zeros(len(dates))


def _run(ids, jobs, cutoff, cfg):
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=cfg['features']['horizon'])
    with ProcessPoolExecutor(os.cpu_count()) as ex:  # 318 chuỗi × ~1 giây: song song theo nhân
        preds = list(ex.map(fit_predict, jobs, [dates] * len(jobs), chunksize=4))
    return pd.DataFrame({'unique_id': np.repeat(ids, len(dates)), 'ds': np.tile(dates, len(ids)),
                         'q10': np.nan, 'q50': np.concatenate(preds), 'q90': np.nan})


@lru_cache(maxsize=1)
def _raw(snapshot):
    return clean.load_raw(snapshot)


def prod_inputs(raw, static, cutoff):
    """Chuỗi train của từng unique_id như production: dòng thô của cặp (ma_tinh mới nhất, csyt_id) tới cutoff."""
    raw = raw[raw.csyt_id.isin(static.csyt_id) & (raw.ds <= cutoff)]
    groups = dict(tuple(raw.groupby(['ma_tinh', 'csyt_id'])))
    empty = raw.iloc[:0]
    return [groups.get((str(r.ma_tinh), r.csyt_id), empty)[['ds', r.field]].rename(columns={r.field: 'y'})
            for r in static.itertuples()]


def prod(df, static, cutoff, cfg):
    """cfg['snapshot']: thư mục snapshot (backtest truyền vào), để đọc dữ liệu thô theo (ma_tinh, csyt_id)."""
    return _run(static.unique_id.to_numpy(), prod_inputs(_raw(cfg['snapshot']), static, cutoff), cutoff, cfg)


def cleaned(df, static, cutoff, cfg):
    past = df.loc[df.ds <= cutoff, ['unique_id', 'ds', 'y', 'available_mask']]
    past = past.assign(y=past.y.where(past.available_mask > 0))
    groups = dict(tuple(past.groupby('unique_id')))
    return _run(static.unique_id.to_numpy(), [groups[u] for u in static.unique_id], cutoff, cfg)
