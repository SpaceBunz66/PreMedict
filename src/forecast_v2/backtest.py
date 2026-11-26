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

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from forecast_v2 import calendar_vn, clean, features, metrics
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
    del long, daily  # ~1 GB; train-final là bước tốn RAM nhất (8,9 triệu mẫu ở snapshot 2026-11-25)
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
    manifest = {'model': 'lgbm', 'snapshot': snap.name, 'train_until': f'{cutoff:%Y-%m-%d}',
                'forecast': [f'{fc.ds.min():%Y-%m-%d}', f'{fc.ds.max():%Y-%m-%d}'],
                'csyt': int(st.csyt_id.nunique()), 'series': len(st), 'train_rows': train_rows,
                'seconds': round(seconds), **provenance(), 'backtest_run': runs[-1].name if runs else None}
    save(out, manifest, cfg)


def provenance():
    """Commit code và sha256 lịch lễ, ghi vào manifest của artifact."""
    git = lambda *a: subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {'git_commit': git('rev-parse', 'HEAD'), 'git_dirty': bool(git('status', '--porcelain', '--untracked-files=no')),
            'calendar_sha256': hashlib.sha256(HOLIDAYS_CSV.read_bytes()).hexdigest()}


def save(out, manifest, cfg):
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f'Đã ghi {out}')


def far(snap, cfg, start=None):
    """train-far: dữ liệu thật dừng ở ngày cuối có dữ liệu (ETL ngừng) nên dự báo start → start + horizon − 1 (mặc
    định từ hôm nay) thẳng từ ngày đó: LightGBM như train-final, chỉ đổi dải h thành gap + 1 .. gap + horizon.
    Sai số đo bằng cách làm y hệt lùi 364 ngày (cùng thứ, cùng mùa năm trước), so với naive "cùng kỳ năm trước".
    q10/q90 chưa hiệu chỉnh (hệ số A5 chỉ fit cho h ≤ 30); độ phủ đo ở backtest.
    Lưu artifacts/lgbm_far/<run_id>/: 3 model, forecast.csv, backtest.csv, importance.csv, config.yaml, manifest.json."""
    h = cfg['features']['horizon']
    start = pd.Timestamp(start or pd.Timestamp.today().normalize())
    last = (f := pd.read_parquet(snap / 'features.parquet', columns=['ds', 'y'])).ds[f.y.notna()].max()
    min_h = (start - last).days
    if min_h < 1:
        raise ValueError(f'start {start:%d/%m/%Y} phải sau ngày cuối có dữ liệu {last:%d/%m/%Y}')
    cf = cfg | {'features': cfg['features'] | {'horizon': min_h + h - 1, 'min_h': min_h}}
    out = ROOT / 'artifacts' / 'lgbm_far' / f'{pd.Timestamp.now():%Y%m%d-%H%M%S}'
    out.mkdir(parents=True)
    daily, long, st, last_day = features.build(snap, cf)  # bảng kéo dài tới start + horizon (không ghi đè file)
    assert last_day == last, (last_day, last)
    st = st[st.strict].reset_index(drop=True)  # như load(): chỉ tập chặt
    long = long[long.unique_id.isin(st.unique_id)]

    cut = last - pd.Timedelta(days=364)
    df, fs = features.fold(long, st, daily, cut, cf)
    fc, _ = thin_fallback(model('lgbm')(df, fs, cut, cf), df, cut)
    e = df.loc[df.ds >= cut + pd.Timedelta(days=min_h), ['unique_id', 'ds', 'y', 'available_mask']].merge(
        fc, on=['unique_id', 'ds'], how='left')
    e['naive'] = last_year(df, cut, e)
    e = e[(e.available_mask > 0) & e.q50.notna() & e.naive.notna()].assign(
        set='backtest', scale=lambda d: d.unique_id.map(metrics.mase_scale(df, cut)))
    s = metrics.score(pd.concat([e.assign(model='lgbm_far'),
                                 e.assign(model='cung_ky_nam_truoc', q50=e.naive, q10=np.nan, q90=np.nan)]))
    s.to_csv(out / 'backtest.csv')
    print(s.round(3).to_string())

    df, fs = features.fold(long, st, daily, last, cf)
    t0 = time.perf_counter()
    fc = model('lgbm')(df, fs, last, cf, save_to=out)
    seconds = time.perf_counter() - t0
    assert len(fc) == len(st) * h and fc.q50.notna().all() and fc.ds.min() == start, 'thiếu dự báo cho một số chuỗi'
    train_rows = fc.attrs['train_rows']
    fc.attrs['importance'].sort_values(ascending=False).rename('gain').to_csv(out / 'importance.csv')
    fc, thin = thin_fallback(fc, df, last)
    fc = st[['unique_id', 'ma_tinh', 'csyt_id', 'field']].merge(fc, on='unique_id').drop(columns='unique_id')
    fc.round(2).to_csv(out / 'forecast.csv', index=False)
    b = s.xs('backtest', level='set')
    manifest = {'model': 'lgbm_far', 'snapshot': snap.name, 'train_until': f'{last:%Y-%m-%d}',
                'forecast': [f'{fc.ds.min():%Y-%m-%d}', f'{fc.ds.max():%Y-%m-%d}'], 'gap_days': min_h - 1,
                'cung_ky_nam_truoc': pd.Series(thin).str.split('|').str[1].value_counts().to_dict(),
                'csyt': int(st.csyt_id.nunique()), 'series': len(st), 'train_rows': train_rows,
                'seconds': round(seconds), **provenance(),
                'backtest': {'train_until': f'{cut:%Y-%m-%d}', 'forecast': [f'{e.ds.min():%Y-%m-%d}', f'{e.ds.max():%Y-%m-%d}'],
                             'series': int(e.unique_id.nunique()), 'points': int(b.n.iloc[0]),
                             'wape': round(b.wape['lgbm_far'], 4), 'wape_cung_ky_nam_truoc': round(b.wape['cung_ky_nam_truoc'], 4),
                             'bias': round(b.bias['lgbm_far'], 4), 'cover80': round(b.cover80['lgbm_far'], 4)}}
    save(out, manifest, cf)


def evaluate(snap, cfg, artifact):
    """Chấm forecast.csv của artifact bằng số thật trong snap; không train gì (dữ liệu sau train_until chỉ để đánh giá).

    Điểm dùng được = mask của masked_panel (thiếu ngày làm việc, ramp-up, ngoại lai, tháng gãy ETL) trên snap, trừ thêm
    ngày nạp dở toàn hệ thống: reporting_share < last_day_min_share, ngoài ngày nghỉ và Tết. Như features.fold, lịch sử
    tới train_until (scale MASE, snaive7/mean4 để so) mask lại chỉ từ dữ liệu tới đó. Mọi model chấm trên cùng các điểm.
    Ghi <artifact>/eval_<snapshot>.md và .parquet; trả về bảng metric như backtest.
    """
    man = json.loads((artifact / 'manifest.json').read_text(encoding='utf-8'))
    cutoff, c, fields = pd.Timestamp(man['train_until']), cfg['clean'], cfg['fields']['direct']
    fc = pd.read_csv(artifact / 'forecast.csv', dtype={'ma_tinh': str, 'csyt_id': str}, parse_dates=['ds'])
    fc['unique_id'] = fc.csyt_id + '|' + fc.field
    daily, _ = clean.merge_provinces(clean.load_raw(snap))
    end = clean.last_data_day(daily, c['last_day_min_share'])
    if end <= cutoff:
        raise ValueError(f'{snap.name}: ngày cuối có dữ liệu {end:%d/%m/%Y} không sau train_until {cutoff:%d/%m/%Y}')
    share = clean.reporting_share(daily)

    def panel(until):
        p = features.masked_panel(daily, fc.csyt_id.unique(), until, cfg)
        partial = p.ds.map(share).lt(c['last_day_min_share']) & ~p.off & ~p.tet
        long = pd.concat([pd.DataFrame({'unique_id': p.csyt_id + '|' + f, 'ds': p.ds, 'y': p[f],
                                        'available_mask': (p[f'mask_{f}'] & ~partial).astype('float32')})
                          for f in fields], ignore_index=True)
        return long[long.unique_id.isin(fc.unique_id)], sorted(p.ds[partial].unique())

    hist, _ = panel(cutoff)
    fut, partial = panel(end)
    fut = fut[(fut.ds > cutoff) & (fut.ds <= fc.ds.max())]
    partial = [d for d in partial if fc.ds.min() <= d <= fc.ds.max()]
    df = pd.concat([hist, fut], ignore_index=True).sort_values(['unique_id', 'ds'], ignore_index=True)

    keys = fc[['unique_id', 'ds']].merge(df[df.ds > cutoff], on=['unique_id', 'ds'])  # điểm đã có số thật
    preds = {man['model']: fc[['unique_id', 'ds', 'q10', 'q50', 'q90']],
             **{n: model(n)(df, None, cutoff, cfg) for n in ('snaive7', 'mean4')}}
    e = pd.concat([keys.merge(p, on=['unique_id', 'ds'], how='left').assign(model=n) for n, p in preds.items()],
                  ignore_index=True)
    cal = calendar_vn.features(pd.date_range(e.ds.min(), e.ds.max()))[['tet_offset', 'is_holiday']]
    info = fc.drop_duplicates('unique_id').set_index('unique_id')[['ma_tinh', 'csyt_id', 'field']]
    tuyen = pd.read_parquet(snap / 'facilities.parquet').drop_duplicates('ma_bv', keep='last').set_index('ma_bv').tuyen_bv
    e = e.join(cal, on='ds').join(info, on='unique_id').assign(
        set='eval', h=(e.ds - cutoff).dt.days, scale=e.unique_id.map(metrics.mase_scale(df, cutoff)))
    e['tuyen'] = e.csyt_id.map(tuyen.astype('string').str.strip()).fillna('unknown').astype(str)
    ok = (e.available_mask > 0) & e.groupby(['unique_id', 'ds']).q50.transform('count').eq(len(preds))
    m = metrics.report(e[ok])

    name = f'eval_{snap.name}'
    e.to_parquet(artifact / f'{name}.parquet', index=False)
    used = e[ok & (e.model == man['model'])]
    md = [f'# Đánh giá {artifact.parent.name}/{artifact.name} bằng số thật của snapshot {snap.name}', '',
          f'- Model {man["model"]}: train tới {cutoff:%d/%m/%Y} (snapshot {man["snapshot"]}), '
          f'dự báo {fc.ds.min():%d/%m/%Y} → {fc.ds.max():%d/%m/%Y}; số thật tới {end:%d/%m/%Y}',
          f'- Điểm chấm: {len(used):,}/{len(fc):,} điểm dự báo, {used.unique_id.nunique()}/{fc.unique_id.nunique()} chuỗi, '
          f'{used.csyt_id.nunique()}/{fc.csyt_id.nunique()} CSYT. Bỏ: chưa có số {len(fc) - len(keys):,}; '
          f'mask = 0 hoặc baseline không dự báo được {len(keys) - len(used):,}',
          f'- Ngày nạp dở (bỏ cho mọi CSYT): {", ".join(f"{d:%d/%m/%Y}" for d in partial) or "không"}',
          '- snaive7, mean4: tính từ số liệu của snapshot này tới ngày train cuối, để so', '']
    for title, s in m.groupby(level='slice', sort=False):
        md += [f'## {title}', '', '```', s.droplevel('slice').round(3).to_string(), '```', '']
    md += ['## tỉnh', '', '```', metrics.score(e[ok], ['ma_tinh']).round(3).to_string(), '```', '']
    (artifact / f'{name}.md').write_text('\n'.join(md), encoding='utf-8')
    print('\n'.join(md[2:6]))
    print(m.loc['tổng'].round(3).to_string())
    print(f'Đã ghi {artifact / name}.md, .parquet')
    return m


def last_year(df, cutoff, keys):
    """y cùng kỳ năm trước (lùi 364 ngày nên cùng thứ; chỉ ngày mask = 1 tới cutoff) cho các dòng keys [unique_id, ds].
    Ngày thiếu lấy trung bình các ngày cùng thứ của chuỗi trong keys."""
    past = df[(df.ds <= cutoff) & (df.available_mask > 0)].set_index(['unique_id', 'ds']).y
    v = pd.Series(past.reindex(pd.MultiIndex.from_arrays([keys.unique_id, keys.ds - pd.Timedelta(days=364)])).to_numpy(),
                  index=keys.index)
    return v.fillna(v.groupby([keys.unique_id, keys.ds.dt.dayofweek]).transform('mean'))


def thin_fallback(fc, df, cutoff, min_days=14):
    """Dự báo xa: chuỗi có < min_days ngày mask = 1 trong 28 ngày tới gốc thì gốc không đủ tin cậy (ngoai_tru gãy ETL
    02–05/2026: model ra gấp 4–12 lần cùng kỳ) -> q50 = cùng kỳ năm trước, q10/q90 giữ tỉ lệ với q50 của model.
    Chuỗi không có số cùng kỳ giữ dự báo model. Trả về (dự báo, danh sách chuỗi đã thay)."""
    recent = df[(df.ds > cutoff - pd.Timedelta(days=28)) & (df.ds <= cutoff)].groupby('unique_id').available_mask.sum()
    thin = recent.index[recent < min_days]
    sel = fc.unique_id.isin(thin)
    q = ['q10', 'q50', 'q90']
    k = (last_year(df, cutoff, fc[sel]) / fc.q50[sel]).where(fc.q50[sel] > 0).fillna(1)
    fc = fc.copy()
    fc.loc[sel, q] = fc.loc[sel, q].mul(k, axis=0)
    return fc, sorted(fc.unique_id[sel].unique())


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
