"""So một DB với snapshot đã train, chỉ đọc (kết nối qua extract.connect: giao dịch READ ONLY).

python scripts/check_db.py [--env .env.prod] [--snapshot 2026-11-18]
"""
import argparse
import os
from contextlib import closing
from datetime import date, timedelta

import pandas as pd

from forecast_v2 import extract
from forecast_v2.extract import DATA_DIR, PREDICT_FIELDS

p = argparse.ArgumentParser()
p.add_argument('--env', default='.env.prod', help='file .env của DB cần kiểm tra (trong forecast_v2/)')
p.add_argument('--snapshot', default=max(d.name for d in DATA_DIR.iterdir() if d.name[:2] == '20'))
a = p.parse_args()

extract.use_env(a.env)
print(f"DB: {os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}  |  snapshot: {a.snapshot}\n")

today = date.today()
neg = ' OR '.join(f'{f} < 0' for f in PREDICT_FIELDS)
num = lambda df: df.apply(pd.to_numeric)  # numeric của Postgres về Decimal
ymd = lambda d: int(d.strftime('%Y%m%d'))

with closing(extract.connect(timeout_ms=600_000)) as conn, conn.cursor() as cur:
    tables = ['hoc.dtm_hoc_kcb_theo_ngay', 'hoc.dm_coso_kcb', 'hoc.config_tinh_dtm_kcb', 'hoc.kcb_tong_hop']
    cur.execute('SELECT ' + ', '.join('to_regclass(%s) IS NOT NULL' for _ in tables), tables)
    print('Bảng:', dict(zip(tables, cur.fetchone())))

    cur.execute(f"""SELECT time_id / 100 AS ym,
            count(DISTINCT csyt_id) FILTER (WHERE tong > 0) AS csyt,
            sum(tong) FILTER (WHERE csyt_id IS NOT NULL) AS tong,
            sum(luot_kham) FILTER (WHERE csyt_id IS NOT NULL) AS luot_kham,
            max(time_id) FILTER (WHERE tong > 0) AS last_day,
            count(*) FILTER (WHERE csyt_id IS NULL) AS null_csyt,
            count(*) FILTER (WHERE {neg}) AS am,
            count(*) FILTER (WHERE tong > 0 AND time_id > %s) AS tuong_lai
        FROM hoc.dtm_hoc_kcb_theo_ngay GROUP BY 1 ORDER BY 1""", (ymd(today),))
    db = num(extract._frame(cur)).set_index('ym')

    cur.execute('''SELECT time_id, count(DISTINCT csyt_id) FILTER (WHERE tong > 0) AS csyt, sum(tong) AS tong
        FROM hoc.dtm_hoc_kcb_theo_ngay WHERE csyt_id IS NOT NULL AND time_id >= %s
        GROUP BY 1 HAVING sum(tong) > 0 ORDER BY 1''', (ymd(today - timedelta(days=30)),))
    recent = num(extract._frame(cur))

    cur.execute('SELECT count(*), count(DISTINCT ma_bv) FROM hoc.dm_coso_kcb')
    fac = cur.fetchone()
    cur.execute('SELECT count(*) FROM hoc.config_tinh_dtm_kcb')
    prov = cur.fetchone()[0]

snap = DATA_DIR / a.snapshot
old = pd.concat(pd.read_parquet(f, columns=['csyt_id', 'time_id', 'tong', 'luot_kham'])
                for f in sorted(snap.glob('daily_*.parquet')))
old['ym'] = old.time_id // 100
s = old.groupby('ym')[['tong', 'luot_kham']].sum().assign(csyt=old[old.tong > 0].groupby('ym').csyt_id.nunique())

t = db.join(s.add_suffix('_snap'), how='outer')
t['lech_tong_%'] = (100 * (t.tong / t.tong_snap - 1)).round(2)
t['lech_csyt'] = t.csyt - t.csyt_snap
pd.set_option('display.width', 200, 'display.max_rows', 500)
print('\nTheo tháng (DB vs snapshot):')
print(t[['csyt', 'csyt_snap', 'lech_csyt', 'tong', 'tong_snap', 'lech_tong_%', 'last_day', 'tuong_lai', 'am']]
      .to_string(float_format='{:,.0f}'.format, formatters={'lech_tong_%': '{:.2f}'.format}))

past = recent[recent.time_id <= ymd(today)]
print(f'\nNgày cuối có tong > 0 (<= hôm nay): {past.time_id.max() if len(past) else "không có trong 30 ngày qua"}')
print(f'Dòng tong > 0 ở ngày tương lai: {int(t.tuong_lai.sum()):,} | csyt_id NULL: {int(t.null_csyt.sum()):,}'
      f' | có giá trị âm: {int(t.am.sum()):,}')
overlap = t.dropna(subset=['tong', 'tong_snap'])
print(f'Tháng trùng với snapshot: {len(overlap)} | lệch tong > 1%: {int((overlap["lech_tong_%"].abs() > 1).sum())}'
      f' | khác số CSYT: {int((overlap.lech_csyt != 0).sum())}')
print(f'dm_coso_kcb: {fac[0]:,} dòng / {fac[1]:,} ma_bv (snapshot {len(pd.read_parquet(snap / "facilities.parquet")):,} dòng)'
      f' | config_tinh_dtm_kcb: {prov} (snapshot {len(pd.read_parquet(snap / "provinces_config.parquet"))})')
print('\nTừ 30 ngày trước (gồm cả ngày tương lai có số):')
print(recent.to_string(index=False) if len(recent) else 'không có dòng nào')
