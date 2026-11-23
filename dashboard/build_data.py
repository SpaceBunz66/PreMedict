"""Dữ liệu cho dashboard demo LightGBM (trong forecast_v2/, PYTHONPATH=src): python dashboard/build_data.py [artifact] -> dashboard/data.js

Artifact mặc định: bản mới nhất của artifacts/lgbm_far (train-far, dự báo từ hôm nay dù dữ liệu thật dừng sớm hơn);
truyền artifacts/lgbm/<run> để xem dự báo 30 ngày ngay sau ngày cuối có dữ liệu (train-final).
Thực tế: daily_2025/2026.parquet của snapshot trong manifest, chỉ các CSYT model dự báo, cộng qua ma_tinh như lúc train.
Chỉ tiêu có dự báo: thực tế chỉ tính trên CSYT có dự báo chỉ tiêu đó (so cùng tập). Dự báo: forecast.csv (q10/q50/q90).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from forecast_v2.clean import merge_provinces

HERE = Path(__file__).parent
ROOT = HERE.parent
START = '2025-01-01'
FIELDS = ['tong', 'luot_vao', 'luot_ra', 'luot_kham', 'noi_tru', 'ngoai_tru', 'cap_cuu', 'tu_vong', 'bhyt',
          'dung_tuyen', 'trai_tuyen', 'tn_gt', 'tn_ld', 'tn_dn', 'tn_nd', 'tn_bl', 'tn_tt', 'tn_bg', 'tn_k']
# shortcut: tên tỉnh theo mã sau sáp nhập 07/2025, gõ tay cho 12 tỉnh có trong tập chặt; mã lạ hiện "Tỉnh <mã>".
PROVINCES = {'01': 'Hà Nội', '12': 'Lai Châu', '22': 'Quảng Ninh', '24': 'Bắc Ninh', '40': 'Nghệ An', '42': 'Hà Tĩnh',
             '44': 'Quảng Trị', '46': 'Huế', '66': 'Đắk Lắk', '79': 'TP. Hồ Chí Minh', '80': 'Tây Ninh', '96': 'Cà Mau'}

sys.stdout.reconfigure(encoding='utf-8')  # console Windows mặc định cp1252: lỗi khi in tiếng Việt
art = Path(sys.argv[1]) if len(sys.argv) > 1 else max((ROOT / 'artifacts' / 'lgbm_far').iterdir())
man = json.loads((art / 'manifest.json').read_text(encoding='utf-8'))
snap = ROOT / 'data' / man['snapshot']

fc = pd.read_csv(art / 'forecast.csv', dtype={'ma_tinh': str, 'csyt_id': str}, parse_dates=['ds'])
units = fc[['ma_tinh', 'csyt_id']].drop_duplicates().sort_values(['ma_tinh', 'csyt_id']).reset_index(drop=True)

raw = pd.concat([pd.read_parquet(snap / f'daily_{y}.parquet', columns=['ma_tinh', 'csyt_id', 'time_id', *FIELDS])
                 for y in (2025, 2026)], ignore_index=True)
raw = raw[raw.csyt_id.isin(units.csyt_id)]
raw['ds'] = pd.to_datetime(raw.pop('time_id').astype(str), format='%Y%m%d')
daily, _ = merge_provinces(raw)
dates = pd.date_range(START, man['train_until'])
daily = daily.set_index(['csyt_id', 'ds']).reindex(pd.MultiIndex.from_product([units.csyt_id, dates])).fillna(0)

fdates = pd.date_range(*man['forecast'])
fc = fc.set_index(['csyt_id', 'field', 'ds']).sort_index()
modeled = fc.index.droplevel('ds').unique()

actual, forecast = [], []
for c in units.csyt_id:
    d = daily.loc[c]
    actual.append({f: d[f].round().astype(int).tolist() for f in FIELDS
                   if f not in fc.index.unique('field') or (c, f) in modeled})
    forecast.append({f: {q: fc.loc[(c, f)].reindex(fdates)[q].round(1).tolist() for q in ('q10', 'q50', 'q90')}
                     for f in fc.index.unique('field') if (c, f) in modeled})

names = pd.read_parquet(snap / 'facilities.parquet', columns=['ma_bv', 'ten_bv']).drop_duplicates('ma_bv')
names = names.set_index('ma_bv').ten_bv
out = {
    'meta': man | {'artifact': art.name},
    'dates': dates.strftime('%Y-%m-%d').tolist(),
    'fdates': fdates.strftime('%Y-%m-%d').tolist(),
    'provinces': {m: PROVINCES.get(m, f'Tỉnh {m}') for m in units.ma_tinh.unique()},
    'units': [{'id': c, 'tinh': m, 'name': str(names.get(c, c))} for m, c in units.itertuples(index=False)],
    'actual': actual,
    'forecast': forecast,
}
assert len(out['units']) == len(actual) == len(forecast) == man['csyt']
(HERE / 'data.js').write_text('window.DATA = ' + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + ';',
                              encoding='utf-8')
print(f"data.js: {len(units)} CSYT, thực tế {START} → {man['train_until']}, dự báo {man['forecast'][0]} → "
      f"{man['forecast'][1]} ({art.name}), {(HERE / 'data.js').stat().st_size / 1e6:.1f} MB")
