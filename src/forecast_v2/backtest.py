"""P7: backtest rolling-origin trên tập chặt (plan §P7). Mỗi model refit tại từng gốc, chỉ qua features.fold.

Đầu ra trong runs/<run_id>/: forecasts.parquet (mọi điểm, kèm y và mask), metrics.parquet, metrics.md,
importance.parquet (model có feature importance).
"""
import hashlib
import importlib
import json
import shutil
import subprocess
import time

import pandas as pd
import yaml
from tqdm import tqdm

from forecast_v2 import clean, features, metrics
from forecast_v2.calendar_vn import HOLIDAYS_CSV
from forecast_v2.extract import ROOT

MODELS = {'snaive7': 'baselines.snaive7', 'mean4': 'baselines.mean4', 'lgbm': 'lgbm.forecast',
          'prophet_prod': 'prophet_prod.prod', 'prophet_clean': 'prophet_prod.cleaned',
          'tide': 'nf_models.tide'}  # trong forecast_v2.models
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
    bar = tqdm(total=len(cuts) * len(names), desc='backtest', unit='model×gốc')
    for cutoff, kind in sorted(cuts.items()):
        df, fs = features.fold(long, st, daily, cutoff, cfg)
        actual = df.loc[df.ds > cutoff, EVAL]
        scale = metrics.mase_scale(df, cutoff)
        for name in names:
            bar.set_postfix_str(f'{cutoff:%d/%m/%Y} {name}')
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
            bar.update()
        tqdm.write(f'{cutoff:%d/%m/%Y} ({kind}): {df.unique_id.nunique()} chuỗi; '
                   + ', '.join(f'{t["model"]} {t["seconds"]:.0f}s' for t in timing[-len(names):]))
    bar.close()
    e = pd.concat(out, ignore_index=True).merge(st[['unique_id', 'field', 'tuyen', 'ma_tinh']], on='unique_id')
    return e, pd.DataFrame(timing), pd.concat(imp, ignore_index=True) if imp else None


def load(snap, cfg, smoke=False):
    """(static tập chặt, bảng long, daily thô). smoke: 20 CSYT, 2 chỉ tiêu (backtest.smoke)."""
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
    return st, long, daily


def latest_runs(name):
    """Các run backtest chính (không drop, tag, smoke) đã xong, có model name, cũ tới mới. Tên run: <giờ>_<model-model…>,
    mọi hậu tố bắt đầu bằng '_' nên làm phần sau '_' đầu tiên không còn là danh sách model."""
    return sorted(p for p in (ROOT / 'runs').glob('*_*') if (p / 'metrics.md').exists()
                  and name in (m := p.name.split('_', 1)[1].split('-')) and set(m) <= set(MODELS))


def calibration(names):
    """A5: hệ số khoảng tin cậy theo nhóm h, fit trên các gốc validation của run mới nhất có từng model; ghi
    interval_factors_<model>.csv vào run đó và in độ phủ 80% trước/sau theo tập × nhóm h (test chỉ để báo cáo)."""
    for name in names:
        run = latest_runs(name)[-1]
        e = pd.read_parquet(run / 'forecasts.parquet')
        e = e[(e.model == name) & (e.available_mask > 0) & e.q50.notna()]
        if e.q10.isna().all():
            print(f'{name}: không có khoảng tin cậy, bỏ qua')
            continue
        f = metrics.interval_factors(e[e.set == 'val'])
        f.to_csv(run / f'interval_factors_{name}.csv')  # một run có thể chứa nhiều model
        h = pd.cut(e.h, metrics.H_BINS, labels=metrics.H_LABELS).rename('h')
        cover = lambda d: d.y.between(d.q10, d.q90).groupby([d.set, h], observed=True).mean()
        print(f'{name} ({run.name}): hệ số\n{f.round(3).to_string()}\n'
              + pd.DataFrame({'trước': cover(e), 'sau': cover(metrics.calibrate(e, f))}).round(3).to_string())
        print(f'Đã ghi {run / f"interval_factors_{name}.csv"}')


def final(snap, cfg, names):
    """train-final (plan §P8): train trên toàn bộ dữ liệu tới ngày cuối có dữ liệu, dự báo horizon ngày sau đó,
    lưu artifacts/<model>/<run_id>/: model, forecast.csv, importance.csv, config.yaml, manifest.json, metric backtest."""
    if names != ['lgbm']:
        raise ValueError('train-final: hiện chỉ lưu được model lgbm (--models lgbm)')
    st, long, daily = load(snap, cfg)
    cutoff = long.ds[long.y.notna()].max()
    df, fs = features.fold(long, st, daily, cutoff, cfg)
    out = ROOT / 'artifacts' / 'lgbm' / f'{pd.Timestamp.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True)
    t0 = time.perf_counter()
    fc = model('lgbm')(df, fs, cutoff, cfg, save_to=out)
    seconds = time.perf_counter() - t0
    h = cfg['features']['horizon']
    assert len(fc) == len(st) * h and fc.q50.notna().all(), 'thiếu dự báo cho một số chuỗi'
    train_rows = fc.attrs['train_rows']  # merge bên dưới làm mất attrs
    fc.attrs['importance'].sort_values(ascending=False).rename('gain').to_csv(out / 'importance.csv')
    runs = latest_runs('lgbm')
    factors = runs[-1] / 'interval_factors_lgbm.csv' if runs else None
    if factors and factors.exists():  # A5: khoảng tin cậy hiệu chỉnh trên validation của run backtest
        shutil.copy(factors, out / factors.name)
        fc = metrics.calibrate(fc.assign(model='lgbm', h=(fc.ds - cutoff).dt.days),
                               pd.read_csv(factors, index_col=[0, 1])).drop(columns=['model', 'h'])
    else:
        print('CẢNH BÁO: chưa có interval_factors_lgbm.csv (lệnh calibrate): q10/q90 chưa hiệu chỉnh')
    fc = st[['unique_id', 'ma_tinh', 'csyt_id', 'field']].merge(fc, on='unique_id').drop(columns='unique_id')
    fc.round(2).to_csv(out / 'forecast.csv', index=False)

    if runs:
        shutil.copy(runs[-1] / 'metrics.md', out / 'backtest_metrics.md')
    git = lambda *a: subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    manifest = {'model': 'lgbm', 'snapshot': snap.name, 'train_until': f'{cutoff:%Y-%m-%d}',
                'forecast': [f'{fc.ds.min():%Y-%m-%d}', f'{fc.ds.max():%Y-%m-%d}'],
                'csyt': int(st.csyt_id.nunique()), 'series': len(st), 'train_rows': train_rows,
                'seconds': round(seconds), 'git_commit': git('rev-parse', 'HEAD'),
                'git_dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
                'calendar_sha256': hashlib.sha256(HOLIDAYS_CSV.read_bytes()).hexdigest(),
                'backtest_run': runs[-1].name if runs else None}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Đã ghi {out}')


def run(snap, cfg, names, smoke=False, tag=''):
    unknown = set(names) - set(MODELS)
    if unknown:
        raise ValueError(f'Model chưa có: {sorted(unknown)}; có: {sorted(MODELS)}')
    drop = cfg['features'].get('drop')  # ablation: tên run không khớp latest_runs nên không lẫn với run chính
    out = ROOT / 'runs' / (f'{pd.Timestamp.now():%Y%m%d-%H%M%S}_{"-".join(names)}'
                           f'{"_drop-" + "-".join(drop) if drop else ""}{"_" + tag if tag else ""}{"_smoke" if smoke else ""}')
    out.mkdir(parents=True)  # tạo trước khi tính: tên sai (vd. tag có ':') báo lỗi ngay, không mất nhiều giờ chạy
    b = cfg['backtest']
    st, long, daily = load(snap, cfg, smoke)

    cuts = cutoffs(b, smoke)
    e, timing, imp = forecasts(long, st, daily, cuts, names, cfg | {'snapshot': snap})  # prophet_prod đọc dữ liệu thô
    if e.q50.isna().any():
        print(f'CẢNH BÁO: {e.q50.isna().sum():,} điểm không có dự báo: {e[e.q50.isna()].groupby("model").size().to_dict()}')
    m = metrics.report(e[(e.available_mask > 0) & e.q50.notna()])

    e.to_parquet(out / 'forecasts.parquet', index=False)
    m.to_parquet(out / 'metrics.parquet')
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    sets = {k: [f'{c:%d/%m/%Y}' for c, v in sorted(cuts.items()) if v == k] for k in ('val', 'test')}
    md = [f'# Backtest {out.name}', '', f'- Snapshot {snap.name}; {st.csyt_id.nunique()} CSYT, {len(st)} chuỗi (tập chặt)',
          f'- Gốc validation: {", ".join(sets["val"]) or "không"}; test: {", ".join(sets["test"]) or "không"}; '
          f'horizon {cfg["features"]["horizon"]} ngày',
          f'- Nhóm feature lgbm bỏ đi: {", ".join(drop) if drop else "không"}',
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
