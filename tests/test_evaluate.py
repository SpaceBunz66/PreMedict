"""evaluate: chấm artifact bằng số thật của snapshot mới, bỏ ngày nạp dở toàn hệ thống; không train gì."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from forecast_v2 import backtest, clean
from forecast_v2.extract import ROOT

DAYS = pd.date_range('2025-01-01', '2025-06-30')
CUTOFF, PARTIAL = pd.Timestamp('2025-05-31'), pd.Timestamp('2025-06-17')  # 17/06/2025: thứ Ba


def test_evaluate_skips_partial_day(tmp_path):
    cfg = yaml.safe_load(open(ROOT / 'configs' / 'base.yaml', encoding='utf-8'))
    level = np.where(DAYS.dayofweek < 5, 100.0, 20.0)
    raw = []
    for i in range(30):
        tong = level.copy()
        tong[DAYS == PARTIAL] *= 0.6 if i < 5 else 0  # nạp dở: chỉ 5/30 CSYT có số, và số còn thiếu
        raw.append(pd.DataFrame({'ma_tinh': '01', 'csyt_id': f'c{i}', 'time_id': DAYS.strftime('%Y%m%d').astype(int),
                                 'tong': tong}))
    raw = pd.concat(raw, ignore_index=True)
    for f in clean.FIELDS[1:]:
        raw[f] = raw.tong / 2
    snap = tmp_path / 'snap'
    snap.mkdir()
    raw.to_parquet(snap / 'daily_2025.parquet', index=False)
    pd.DataFrame({'ma_bv': raw.csyt_id.unique(), 'tuyen_bv': '3'}).to_parquet(snap / 'facilities.parquet', index=False)

    art = tmp_path / 'artifacts' / 'lgbm' / 'run1'
    art.mkdir(parents=True)
    (art / 'manifest.json').write_text(json.dumps({'model': 'lgbm', 'snapshot': 'old', 'train_until': f'{CUTOFF:%Y-%m-%d}'}))
    june = DAYS[DAYS > CUTOFF]
    truth = np.where(june.dayofweek < 5, 100.0, 20.0)  # dự báo đúng mức bình thường, kể cả ngày nạp dở
    fc = pd.concat([pd.DataFrame({'ma_tinh': '01', 'csyt_id': f'c{i}', 'field': f, 'ds': june, 'q50': truth * k})
                    for i in range(3) for f, k in (('tong', 1), ('luot_kham', 0.5))])
    fc.assign(q10=fc.q50 * 0.8, q90=fc.q50 * 1.2).to_csv(art / 'forecast.csv', index=False)

    m = backtest.evaluate(snap, cfg, art)
    tot = m.loc['tổng'].droplevel(['group', 'set'])
    assert tot.wape['lgbm'] == pytest.approx(0) and tot.cover80['lgbm'] == 1  # ngày nạp dở bị bỏ, không thành sai số
    assert tot.wape['snaive7'] == pytest.approx(0) and tot.n['lgbm'] == 3 * 2 * 29
    e = pd.read_parquet(art / 'eval_snap.parquet')
    assert (e[e.ds == PARTIAL].available_mask == 0).all() and (e[e.ds != PARTIAL].available_mask == 1).all()
    assert '17/06/2025' in (art / 'eval_snap.md').read_text(encoding='utf-8')
