"""P5: bảng feature dạng long dùng chung cho mọi model (plan §P5).

Khóa unique_id = csyt_id|chỉ_tiêu, ds. `table` build một lần cho cả snapshot (tầng nới; cột strict trong static).
Mọi thứ suy ra từ target (available_mask, quy mô log, tỉ lệ mở cửa cuối tuần) được `fold` tính lại chỉ từ dữ liệu tới
cutoff: mask của P1 dùng median trượt đặt ở giữa và tỉ lệ mở cửa trên cả lịch sử, nên tính trên toàn bộ dữ liệu là
nhìn thấy tương lai (tests/test_no_leakage.py).

Cột của bảng: y, available_mask; covariate tương lai (lịch, lễ theo tỉnh, clim_*); covariate quá khứ w_* (khí tượng
quan trắc, giá trị tại ds là quan trắc của ds − weather_obs_delay_days).
"""
import numpy as np
import pandas as pd

from forecast_v2 import calendar_vn, clean

CLIM = ['clim_t_mean', 'clim_rh_mean', 'clim_rain_mm']
NOT_OBS = ['ds', 'csyt_id', 'radius_km', 'doy', *CLIM]  # cột của weather_daily không phải biến quan trắc


def masked_panel(daily, ids, end, cfg):
    """Panel (csyt_id, ds) của các CSYT ids, từ ngày hoạt động đầu tiên tới end, kèm y các chỉ tiêu và mask_<chỉ tiêu>
    (plan §P1 "Làm sạch": thiếu ngày làm việc, ramp-up, ngoại lai, tháng gãy ETL). Chỉ dùng dữ liệu tới end.
    daily phải gồm mọi CSYT: tháng gãy ETL xét trên toàn hệ thống.
    """
    c, fields = cfg['clean'], cfg['fields']['direct']
    daily = daily[daily.ds <= end]
    breaks = clean.etl_breaks(daily, [f for f in fields if f != 'tong'], c['etl_break_drop'])
    p = clean.build_panel(daily.loc[daily.csyt_id.isin(ids), ['csyt_id', 'ds', *fields]], end, c)
    month = p.ds.dt.to_period('M')
    for f in fields:
        bad = clean.outliers(p, f, c['outlier_z'])
        if f in breaks:
            bad |= month.map(breaks[f]).eq(True)
        p[f'mask_{f}'] = p.ok & ~bad
    return p


def table(daily, report, static, weather, clim, cfg):
    """(bảng long, static theo unique_id, ngày cuối có dữ liệu). Bảng kéo dài tới ngày cuối + horizon: dòng tương lai
    có y NaN, mask 0. available_mask tính trên toàn bộ dữ liệu, chỉ đúng cho đánh giá; khi train dùng `fold`.
    """
    c, fe, fields = cfg['clean'], cfg['features'], cfg['fields']['direct']
    radius = static.radius_km.rename_axis('csyt_id').reset_index()
    w = weather.merge(radius, on=['csyt_id', 'radius_km'])
    has = w.groupby('csyt_id').t_mean.count() > 0
    ids = has.index[has]  # bỏ CSYT không có khí tượng (vd. 45105 đảo Cồn Cỏ: toàn ô biển)

    last_day = clean.last_data_day(daily, c['last_day_min_share'])
    p = masked_panel(daily, ids, last_day, cfg)
    cols = ['csyt_id', 'ds', *fields, *(f'mask_{f}' for f in fields)]
    future = pd.DataFrame({'csyt_id': np.repeat(ids.to_numpy(), fe['horizon']),
                           'ds': np.tile(pd.date_range(last_day + pd.Timedelta(days=1), periods=fe['horizon']), len(ids))})
    g = pd.concat([p[cols], future], ignore_index=True)

    g = g.join(calendar_vn.features(pd.date_range(g.ds.min(), g.ds.max())), on='ds')
    g['regional_holiday'] = calendar_vn.regional(g.ds, g.csyt_id.map(static.ma_tinh))
    g['doy'] = g.ds.dt.dayofyear
    g = g.merge(clim.merge(radius, on=['csyt_id', 'radius_km'])[['csyt_id', 'doy', *CLIM]], on=['csyt_id', 'doy'],
                how='left').drop(columns='doy')
    obs = w[['csyt_id', 'ds', *(k for k in w if k not in NOT_OBS)]]
    obs = obs.assign(ds=obs.ds + pd.Timedelta(days=fe['weather_obs_delay_days']))
    g = g.merge(obs.rename(columns=lambda k: k if k in ('csyt_id', 'ds') else f'w_{k}'), on=['csyt_id', 'ds'], how='left')

    cov = [k for k in g if k not in cols]
    for k in cov:
        g[k] = g[k].astype('category' if g[k].dtype == object else 'float32')
    use = report.loc[ids, [f'use_{f}' for f in fields]]
    long = pd.concat([g.loc[g.csyt_id.isin(use.index[use[f'use_{f}']]), ['csyt_id', 'ds', f, f'mask_{f}', *cov]]
                      .rename(columns={f: 'y', f'mask_{f}': 'available_mask'}).assign(field=f) for f in fields],
                     ignore_index=True)
    long['available_mask'] = long.available_mask.eq(True).astype('float32')  # dòng tương lai: NaN -> 0
    long.insert(0, 'unique_id', long.csyt_id + '|' + long.field)

    st = long[['unique_id', 'csyt_id', 'field']].drop_duplicates().join(
        static[['ma_tinh', 'latitude', 'longitude', 'tuyen', 'hang', 'strict']], on='csyt_id').reset_index(drop=True)
    st[['ma_tinh', 'tuyen', 'hang']] = st[['ma_tinh', 'tuyen', 'hang']].astype('category')
    return long, st, last_day


def fold(df, static, daily, cutoff, cfg):
    """Dữ liệu cho fold có gốc cutoff: các dòng tới cutoff + horizon.

    - Tới cutoff: available_mask tính lại chỉ từ dữ liệu tới cutoff.
    - Sau cutoff: giữ y và mask tính trên toàn bộ dữ liệu (để đánh giá); xóa w_* (quan trắc chưa có tại cutoff).
    - static thêm log_scale (log1p trung bình y) và weekend_open (tỉ lệ T7/CN có tong > 0), trên 365 ngày tới cutoff.
    Bỏ chuỗi không có ngày dùng được nào tới cutoff. daily: mọi CSYT (xem masked_panel).
    """
    df = df[df.ds <= cutoff + pd.Timedelta(days=cfg['features']['horizon'])].reset_index(drop=True)
    past = df.ds <= cutoff
    p = masked_panel(daily, df.csyt_id.unique(), cutoff, cfg)
    m = pd.concat([p[['csyt_id', 'ds']].assign(field=f, m=p[f'mask_{f}']) for f in cfg['fields']['direct']])
    new = df[['csyt_id', 'ds', 'field']].merge(m, on=['csyt_id', 'ds', 'field'], how='left').m.eq(True)
    df.loc[past, 'available_mask'] = new[past].astype('float32')
    df.loc[~past, [k for k in df if k.startswith('w_')]] = np.nan

    seen = df[past].groupby('unique_id').available_mask.max() > 0
    df = df[df.unique_id.isin(seen.index[seen])].reset_index(drop=True)
    hist = df[(df.ds <= cutoff) & (df.ds > cutoff - pd.Timedelta(days=365)) & (df.available_mask > 0)]
    st = static[static.unique_id.isin(seen.index[seen])].reset_index(drop=True)
    st['log_scale'] = st.unique_id.map(np.log1p(hist.groupby('unique_id').y.mean())).astype('float32')
    wk = hist[(hist.field == 'tong') & (hist.ds.dt.dayofweek >= 5)]
    st['weekend_open'] = st.csyt_id.map((wk.y > 0).groupby(wk.csyt_id).mean()).astype('float32')
    return df, st


def run(snap, cfg):
    daily, _ = clean.merge_provinces(clean.load_raw(snap))
    long, st, last_day = table(daily, pd.read_parquet(snap / 'quality_report.parquet'),
                               pd.read_parquet(snap / 'facilities_static.parquet'),
                               pd.read_parquet(snap / 'weather_daily.parquet'),
                               pd.read_parquet(snap / 'weather_clim.parquet'), cfg)
    long.to_parquet(snap / 'features.parquet', index=False)
    st.to_parquet(snap / 'features_static.parquet', index=False)
    past = long[long.ds <= last_day]
    print(f'{st.csyt_id.nunique()} CSYT ({st[st.strict].csyt_id.nunique()} tập chặt), {len(st)} chuỗi '
          f'{st.field.value_counts().to_dict()}; {len(long):,} dòng × {long.shape[1]} cột tới '
          f'{long.ds.max():%d/%m/%Y} (ngày cuối có dữ liệu {last_day:%d/%m/%Y}); mask = 0 ở {1 - past.available_mask.mean():.1%} '
          f'số dòng quá khứ; thiếu w_t_mean {past.w_t_mean.isna().mean():.2%}. Đã ghi {snap / "features.parquet"}')
