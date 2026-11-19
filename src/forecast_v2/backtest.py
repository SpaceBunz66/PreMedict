"""P7: backtest rolling-origin trên tập chặt (plan §P7). Mỗi model refit tại từng gốc, chỉ qua features.fold.

Đầu ra trong runs/<run_id>/: forecasts.parquet (mọi điểm, kèm y và mask), metrics.parquet, metrics.md,
importance.parquet (model có feature importance).
"""
import importlib
import time

import pandas as pd

from forecast_v2 import clean, features, metrics
from forecast_v2.extract import ROOT

MODELS = {'snaive7': 'baselines.snaive7', 'mean4': 'baselines.mean4', 'lgbm': 'lgbm.forecast',
          'prophet_prod': 'prophet_prod.prod', 'prophet_clean': 'prophet_prod.cleaned'}  # trong forecast_v2.models
EVAL = ['unique_id', 'ds', 'y', 'available_mask', 'tet_offset', 'is_holiday']


def model(name):
    module, fn = MODELS[name].rsplit('.', 1)
    return getattr(importlib.import_module(f'forecast_v2.models.{module}'), fn)  # import khi cần: lightgbm/torch nặng


def cutoffs(b, smoke):
    if smoke:
        return {pd.Timestamp(d): 'test' for d in b['smoke']['cutoffs']}
    return {pd.Timestamp(d): 'val' for d in b['val']} | {pd.Timestamp(d): 'test' for d in b['test']}


def forecasts(long, st, daily, cuts, names, cfg):
    """Dự báo của mọi model tại mọi gốc, nối với y thật, mask và scale MASE.
    Trả về (bảng, thời gian theo model × gốc, feature importance theo model × gốc)."""
    h = pd.Timedelta(days=cfg['features']['horizon'])
    last = long.ds[long.y.notna()].max()
    if max(cuts) + h > last:
        raise ValueError(f'Gốc {max(cuts):%d/%m/%Y} + horizon vượt ngày cuối có dữ liệu {last:%d/%m/%Y}')
    out, timing, imp = [], [], []
    for cutoff, kind in sorted(cuts.items()):
        df, fs = features.fold(long, st, daily, cutoff, cfg)
        actual = df.loc[df.ds > cutoff, EVAL]
        scale = metrics.mase_scale(df, cutoff)
        for name in names:
            t0 = time.perf_counter()
            fc = model(name)(df, fs, cutoff, cfg)
            timing.append({'model': name, 'cutoff': cutoff, 'seconds': time.perf_counter() - t0,
                           'train_rows': fc.attrs.get('train_rows')})
            if 'importance' in fc.attrs:
                gain = fc.attrs['importance']
                imp.append(pd.DataFrame({'feature': gain.index, 'share': gain.to_numpy() / gain.sum(),
                                         'model': name, 'cutoff': cutoff, 'set': kind}))
            fc = actual.merge(fc, on=['unique_id', 'ds'], how='left')  # thiếu dự báo -> NaN, không âm thầm bỏ
            out.append(fc.assign(model=name, set=kind, cutoff=cutoff, h=(fc.ds - cutoff).dt.days,
                                 scale=fc.unique_id.map(scale)))
        print(f'{cutoff:%d/%m/%Y} ({kind}): {df.unique_id.nunique()} chuỗi; '
              + ', '.join(f'{t["model"]} {t["seconds"]:.0f}s' for t in timing[-len(names):]), flush=True)
    e = pd.concat(out, ignore_index=True).merge(st[['unique_id', 'field', 'tuyen', 'ma_tinh']], on='unique_id')
    return e, pd.DataFrame(timing), pd.concat(imp, ignore_index=True) if imp else None


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
    e, timing, imp = forecasts(long, st, daily, cuts, names, cfg | {'snapshot': snap})  # prophet_prod đọc dữ liệu thô
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
    if 'prophet_prod' in names and len(names) > 1:
        v = metrics.versus(e[(e.available_mask > 0) & e.q50.notna()])
        md += ['## So với ProphetProd (Wilcoxon ghép cặp theo chuỗi)', '', '```', v.round(4).to_string(), '```', '']
        print(v.round(4).to_string())
    if imp is not None:
        imp.to_parquet(out / 'importance.parquet', index=False)
        ref = imp[imp.set == 'val'] if (imp.set == 'val').any() else imp  # chọn biến trên validation (plan §P6)
        rank = ref.groupby(['model', 'feature']).share.mean().sort_values(ascending=False)
        for mdl, r in rank.groupby(level='model'):
            r = r.droplevel('model')
            md += [f'## Importance {mdl} (tỉ lệ gain, trung bình các gốc {"validation" if (imp.set == "val").any() else "đã chạy"})', '',
                   '```', r.head(25).round(4).to_string(), '```', '', 'Khí tượng quan trắc (w_*), xếp hạng:', '',
                   '```', r[r.index.str.startswith('w_')].round(4).to_string(), '```', '']
    (out / 'metrics.md').write_text('\n'.join(md), encoding='utf-8')
    print(m.loc['tổng'].round(3).to_string())
    print(f'Đã ghi {out}')
