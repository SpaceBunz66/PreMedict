"""Đối chiếu ProphetProd với kết quả code cũ (chạy bằng .venv-ml, sau old_prophet_call.py).

So sánh (1) dữ liệu code cũ đọc từ DB với snapshot, (2) dự báo: code cũ (prophet 1.1.5, dữ liệu DB) với ProphetProd
trên cùng dữ liệu DB (tách ảnh hưởng phiên bản) và trên snapshot (đường dữ liệu của forecast_v2).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_v2 import clean
from forecast_v2.extract import DATA_DIR
from forecast_v2.models import prophet_prod

OUT = Path(__file__).resolve().parents[1] / "runs" / "prophet_crosscheck"
res = json.loads((OUT / 'old_result.json').read_text(encoding='utf-8'))
db = pd.read_csv(OUT / 'old_input.csv', parse_dates=['ds'])
dates = pd.to_datetime(res['forecast_dates'])
old = pd.DataFrame(res['forecast_rows']).assign(ds=dates)
call_day = pd.Timestamp(res['last_data_date']) + pd.Timedelta(days=1)  # time_id < hôm nay lúc gọi

raw = clean.load_raw(max(DATA_DIR.glob('20??-??-??')))
snap = raw[(raw.ma_tinh == res['ma_tinh']) & (raw.csyt_id == res['csyt_id']) & (raw.ds < call_day)]
snap = snap.set_index('ds').sort_index()
d = db.set_index('ds').sort_index()
print(f'Dòng: DB {len(d)}, snapshot {len(snap)}; ngày chỉ có ở DB: {len(d.index.difference(snap.index))}, '
      f'chỉ có ở snapshot: {len(snap.index.difference(d.index))}')

rows = []
for f in clean.FIELDS:
    both = d[[f]].join(snap[[f]], rsuffix='_snap', how='inner')
    a = prophet_prod.fit_predict(d[[f]].reset_index().rename(columns={f: 'y'}), dates)
    b = prophet_prod.fit_predict(snap[[f]].reset_index().rename(columns={f: 'y'}), dates)
    o = old[f].to_numpy(float)
    rel = lambda x: np.abs(x - o).sum() / max(np.abs(o).sum(), 1e-9)
    rows.append({'field': f, 'data_max_diff': float((both[f] - both[f + '_snap']).abs().max()),
                 'old_sum': o.sum(), 'new_on_db_rel': rel(a), 'new_on_snapshot_rel': rel(b),
                 'new_on_db_max_abs': np.abs(a - o).max()})
print(pd.DataFrame(rows).set_index('field').round(4).to_string())
