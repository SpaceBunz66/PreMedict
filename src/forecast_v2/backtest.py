"""P7: backtest rolling-origin trên tập chặt (plan §P7). Mỗi model refit tại từng gốc, chỉ qua features.fold.

Đầu ra trong runs/<run_id>/: forecasts.parquet (mọi điểm, kèm y và mask), metrics.parquet, metrics.md.
"""
import importlib
import time

import pandas as pd

from forecast_v2 import clean, features, metrics
from forecast_v2.extract import ROOT

MODELS = {'snaive7': 'baselines', 'mean4': 'baselines'}  # tên model -> module trong forecast_v2.models
EVAL = ['unique_id', 'ds', 'y', 'available_mask', 'tet_offset', 'is_holiday']


def model(name):
    return getattr(importlib.import_module(f'forecast_v2.models.{MODELS[name]}'), name)  # import chậm: torch nặng


def cutoffs(b, smoke):
    if smoke:
        return {pd.Timestamp(d): 'test' for d in b['smoke']['cutoffs']}
    return {pd.Timestamp(d): 'val' for d in b['val']} | {pd.Timestamp(d): 'test' for d in b['test']}


def forecasts(long, st, daily, cuts, names, cfg):
    """Dự báo của mọi model tại mọi gốc, nối với y thật, mask và scale MASE. Trả về (bảng, thời gian theo model × gốc)."""
    h = pd.Timedelta(days=cfg['features']['horizon'])
    last = long.ds[long.y.notna()].max()
    if max(cuts) + h > last:
        raise ValueError(f'Gốc {max(cuts):%d/%m/%Y} + horizon vượt ngày cuối có dữ liệu {last:%d/%m/%Y}')
    out, timing = [], []
    for cutoff, kind in sorted(cuts.items()):
        df, fs = features.fold(long, st, daily, cutoff, cfg)
        actual = df.loc[df.ds > cutoff, EVAL]
        scale = metrics.mase_scale(df, cutoff)
        for name in names:
            t0 = time.perf_counter()
            fc = model(name)(df, fs, cutoff, cfg)
            timing.append({'model': name, 'cutoff': cutoff, 'seconds': time.perf_counter() - t0})
            fc = actual.merge(fc, on=['unique_id', 'ds'], how='left')  # thiếu dự báo -> NaN, không âm thầm bỏ
            out.append(fc.assign(model=name, set=kind, cutoff=cutoff, h=(fc.ds - cutoff).dt.days,
                                 scale=fc.unique_id.map(scale)))
        print(f'{cutoff:%d/%m/%Y} ({kind}): {df.unique_id.nunique()} chuỗi; '
              + ', '.join(f'{t["model"]} {t["seconds"]:.0f}s' for t in timing[-len(names):]), flush=True)
    e = pd.concat(out, ignore_index=True).merge(st[['unique_id', 'field', 'tuyen', 'ma_tinh']], on='unique_id')
    return e, pd.DataFrame(timing)


def run(snap, cfg, names, smoke=False):
    unknown = set(names) - set(MODELS)
    if unknown:
        raise ValueError(f'Model chưa có: {sorted(unknown)}; có: {sorted(MODELS)}')
    b = cfg['backtest']
    st = pd.read_parquet(snap / 'features_static.parquet')
    st = st[st.strict]  # quyết định 19/11/2026: chỉ tập chặt
    if smoke:
        ids = sorted(st.csyt_id.unique())
        ids = ids[::max(1, len(ids) // b['smoke']['csyt'])][:b['smoke']['csyt']]  # rải đều theo mã
        st = st[st.csyt_id.isin(ids) & st.field.isin(b['smoke']['fields'])]
    long = pd.read_parquet(snap / 'features.parquet')
    long = long[long.unique_id.isin(st.unique_id)]
    daily, _ = clean.merge_provinces(clean.load_raw(snap))  # mọi CSYT: fold xét tháng gãy ETL trên toàn hệ thống

    cuts = cutoffs(b, smoke)
    e, timing = forecasts(long, st, daily, cuts, names, cfg)
    if e.q50.isna().any():
        print(f'CẢNH BÁO: {e.q50.isna().sum():,} điểm không có dự báo: {e[e.q50.isna()].groupby("model").size().to_dict()}')
    m = metrics.report(e[(e.available_mask > 0) & e.q50.notna()])

    out = ROOT / 'runs' / f'{pd.Timestamp.now():%Y%m%d-%H%M%S}_{"-".join(names)}{"_smoke" if smoke else ""}'
    out.mkdir(parents=True)
    e.to_parquet(out / 'forecasts.parquet', index=False)
    m.to_parquet(out / 'metrics.parquet')
    sets = {k: [f'{c:%d/%m/%Y}' for c, v in sorted(cuts.items()) if v == k] for k in ('val', 'test')}
    md = [f'# Backtest {out.name}', '', f'- Snapshot {snap.name}; {st.csyt_id.nunique()} CSYT, {len(st)} chuỗi (tập chặt)',
          f'- Gốc validation: {", ".join(sets["val"]) or "không"}; test: {", ".join(sets["test"]) or "không"}; '
          f'horizon {cfg["features"]["horizon"]} ngày',
          f'- Thời gian (giây/gốc, trung bình): {timing.groupby("model").seconds.mean().round(1).to_dict()}', '']
    for name, s in m.groupby(level='slice', sort=False):
        md += [f'## {name}', '', '```', s.droplevel('slice').round(3).to_string(), '```', '']
    (out / 'metrics.md').write_text('\n'.join(md), encoding='utf-8')
    print(m.loc['tổng'].round(3).to_string())
    print(f'Đã ghi {out}')
