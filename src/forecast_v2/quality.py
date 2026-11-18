"""P1: chấm chất lượng từng CSYT và chọn tập train (plan §P1). Xuất quality_report.parquet và audit.md."""
import numpy as np
import pandas as pd

from forecast_v2 import clean


def metrics(panel, last_day, fields, q):
    """Chỉ số theo CSYT, tính trên window_days ngày cuối tới last_day."""
    m = pd.DataFrame({'first_day': panel.groupby('csyt_id').ds.min(),
                      'last_active': panel[panel.tong > 0].groupby('csyt_id').ds.max()})
    m['history_days'] = (last_day - m.first_day).dt.days + 1
    m['days_since_last'] = (last_day - m.last_active).dt.days

    w = panel[panel.ds > last_day - pd.Timedelta(days=q['window_days'])].copy()
    w['month'] = w.ds.dt.to_period('M')
    n_days = w.groupby('month').ds.nunique()
    w['full'] = w.month.isin(n_days.index[n_days.to_numpy() == n_days.index.days_in_month])  # tháng trọn trong cửa sổ
    wk = w[(w.ds.dt.dayofweek < 5) & ~w.off & ~w.tet]  # ngày làm việc, bỏ cửa sổ Tết
    zero = wk.tong == 0
    m['workday_coverage'] = 1 - zero.groupby(wk.csyt_id).mean()
    run = (~zero).groupby(wk.csyt_id).cumsum()  # các ngày 0 liên tiếp có cùng run
    m['max_workday_gap'] = zero.groupby([wk.csyt_id, run]).sum().groupby(level=0).max()

    monthly = w[w.full].groupby(['csyt_id', 'month'])[fields].sum()
    m = m.join(monthly.groupby(level=0).median().add_prefix('month_median_'))
    m['year_total'] = w.groupby('csyt_id').tong.sum()
    t = monthly.tong[~monthly.index.get_level_values('month').isin(w.month[w.tet].unique())]  # bỏ tháng Tết
    m['min_month_ratio'] = (t / m.month_median_tong.reindex(t.index.get_level_values(0)).to_numpy()).groupby(level=0).min()
    m['outlier_ratio'] = w.groupby('csyt_id').outlier.sum() / w.groupby('csyt_id').ok.sum()

    level = wk[wk.full].groupby(['csyt_id', 'month']).tong.median().unstack()  # CSYT × tháng
    before = level.T.rolling(3).median().T  # median 3 tháng tính tới tháng k
    shift = before.shift(-3, axis=1) / before  # 3 tháng sau k so với 3 tháng tới k
    m['shift_up'], m['shift_down'] = shift.max(axis=1), shift.min(axis=1)

    act = w[w.tong > 0]
    m['bhyt_split_mismatch'] = (act.dung_tuyen + act.trai_tuyen != act.bhyt).groupby(act.csyt_id).mean()  # chỉ báo cáo
    return m


def checks(m, t):
    """Mỗi cột là một tiêu chí của plan §P1, theo thứ tự funnel; True = đạt. NaN (thiếu số liệu) là không đạt."""
    return pd.DataFrame({
        'active': m.days_since_last <= t['max_days_since_last'],
        'history': m.history_days >= t['min_history_days'],
        'coords': m.coords_ok,
        'coverage': m.workday_coverage >= t['min_workday_coverage'],
        'gap': m.max_workday_gap <= t['max_workday_gap'],
        'volume': (m.month_median_tong >= t['min_month_median']) & (m.year_total >= t['min_year_total']),
        'collapse': m.min_month_ratio >= t['min_month_ratio'],
        'outliers': m.outlier_ratio <= t['max_outlier_ratio'],
        'level_shift': (m.shift_up <= t['level_shift_up']) & (m.shift_down >= t['level_shift_down']),
    })


def neighbor_share(lat, lon, tinh, k):
    """Tỉ lệ trong k CSYT gần nhất (trừ chính nó) có cùng mã tỉnh. 0 = tọa độ nằm lọt trong vùng của tỉnh khác."""
    lat, lon, tinh = (np.asarray(a) for a in (lat, lon, tinh))
    k = min(k, len(lat) - 1)
    if k < 1:
        return np.full(len(lat), np.nan)
    out = []
    for i in range(0, len(lat), 1000):  # theo khối: ma trận khoảng cách 1000 × n
        j = slice(i, i + 1000)
        d = (lat[j, None] - lat) ** 2 + ((lon[j, None] - lon) * np.cos(np.radians(lat[j, None]))) ** 2
        d[np.arange(len(d)), np.arange(i, i + len(d))] = np.inf
        out.append((tinh[np.argpartition(d, k - 1, axis=1)[:, :k]] == tinh[j, None]).mean(axis=1))
    return np.concatenate(out)


def _facilities(fac, q):
    """Một dòng cho mỗi ma_bv, ưu tiên dòng có tọa độ hợp lệ và có tuyến.

    Tọa độ hợp lệ: trong khung lat/lon, không phải tọa độ điền tạm (cả hai là bội của 0,1°, vd. (21,0; 105,0) dùng chung
    cho 34 CSYT), và có ít nhất 1 CSYT cùng tỉnh trong coord_neighbors CSYT gần nhất của dm_coso_kcb.
    """
    fac = fac.assign(ma_bv=fac.ma_bv.astype(str))
    inrange = fac.latitude.between(*q['lat']) & fac.longitude.between(*q['lon'])
    tenth = lambda x: (x * 10).round(6) % 1 == 0
    placeholder = tenth(fac.latitude) & tenth(fac.longitude)
    ref = inrange & ~placeholder & fac.ma_tinh.notna()
    fac['province_share'] = np.nan  # NaN: không xét được (thiếu tọa độ/mã tỉnh)
    fac.loc[ref, 'province_share'] = neighbor_share(fac.latitude[ref], fac.longitude[ref],
                                                    fac.ma_tinh[ref].astype(str), q['coord_neighbors'])
    fac['coords_ok'] = (inrange & ~placeholder & fac.province_share.ne(0)
                        & fac.tuyen_bv.fillna('').str.strip().ne('')).astype(bool)
    return (fac.sort_values('coords_ok', ascending=False).drop_duplicates('ma_bv').set_index('ma_bv')
            [['ten_bv', 'tuyen_bv', 'hang_bv', 'latitude', 'longitude', 'province_share', 'coords_ok']])


def evaluate(raw, facilities, cfg):
    """raw: dòng (ma_tinh, csyt_id, ds, chỉ tiêu...). Trả về (report theo CSYT, ngày cuối có dữ liệu, panel)."""
    c, q, fields = cfg['clean'], cfg['quality'], cfg['fields']
    daily, province = clean.merge_provinces(raw)
    last_day = clean.last_data_day(daily, c['last_day_min_share'])
    panel = clean.build_panel(daily, last_day, c)
    panel['outlier'] = clean.outliers(panel, 'tong', c['outlier_z'])

    m = metrics(panel, last_day, fields['direct'], q)
    m['ma_tinh'] = province
    m = m.join(_facilities(facilities, q))
    m['coords_ok'] = m.coords_ok.eq(True)  # CSYT không có trong dm_coso_kcb: NaN -> False
    strict = checks(m, q)
    m['failed'] = strict.apply(lambda r: ','.join(r.index[~r]), axis=1)
    m['pass_strict'] = strict.all(axis=1)
    m['pass_relaxed'] = checks(m, q | q['relaxed']).all(axis=1)
    for f in fields['direct']:
        m[f'use_{f}'] = m[f'month_median_{f}'] >= fields['min_month_median']
    return m, last_day, panel


def audit(snap, raw, last_day, report, breaks, cfg):
    """audit.md: các con số khảo sát (plan §1) và funnel chọn tập train, để kiểm tra lại mỗi lần refresh dữ liệu."""
    q, direct = cfg['quality'], cfg['fields']['direct']
    active = raw[raw.tong > 0]
    pairs = raw.groupby(['ma_tinh', 'csyt_id']).tong.sum()
    after = raw[raw.ds > last_day]
    province_last = active.groupby('ma_tinh').ds.max()
    stopped = province_last[province_last < last_day - pd.Timedelta(days=q['max_days_since_last'])]
    strict = report[report.pass_strict]
    lines = [
        f'# Audit dữ liệu, snapshot {snap.name}', '',
        '## Khảo sát (plan §1)', '',
        f'- Dòng (ma_tinh × CSYT × ngày): {len(raw):,}; tỉnh: {raw.ma_tinh.nunique()}; CSYT: {raw.csyt_id.nunique():,}; '
        f'từ {raw.ds.min():%d/%m/%Y} tới {raw.ds.max():%d/%m/%Y}',
        f'- Cặp (ma_tinh, CSYT) toàn 0: {(pairs == 0).sum():,}/{len(pairs):,}',
        f'- **Ngày cuối có dữ liệu: {last_day:%d/%m/%Y}**. Sau ngày này: {len(after):,} dòng, '
        f'trong đó {(after.tong > 0).sum():,} dòng có tong > 0 (bị bỏ)',
        f'- Tỉnh dừng nạp sớm (ngày cuối có tong > 0): '
        + (', '.join(f'{p} ({d:%d/%m/%Y})' for p, d in stopped.sort_values().items()) or 'không có'),
        f'- CSYT có dữ liệu dưới > 1 ma_tinh (sáp nhập tỉnh): {(active.groupby("csyt_id").ma_tinh.nunique() > 1).sum():,}',
        f'- luot_ra = tong trên {(active.luot_ra == active.tong).mean():.2%} số dòng có tong > 0',
        '- Tháng gãy ETL (tỉ lệ so với tong): '
        + ('; '.join(f'{f}: {", ".join(str(p) for p in breaks.index[breaks[f]])}' for f in breaks if breaks[f].any())
           or 'không có'),
        '', f'## Tập train (plan §P1), cửa sổ {q["window_days"]} ngày tới {last_day:%d/%m/%Y}', '',
        '| Tiêu chí | Còn lại (cộng dồn) | Không đạt tiêu chí này |', '|---|---:|---:|',
        f'| (CSYT có dữ liệu) | {len(report):,} | |',
    ]
    ok = checks(report, q)
    keep = pd.Series(True, index=report.index)
    for name in ok:
        keep &= ok[name]
        lines.append(f'| {name} | {keep.sum():,} | {(~ok[name]).sum():,} |')
    lines += [
        '',
        f'- **Tập chặt: {len(strict)} CSYT**; tập nới (coverage ≥ {q["relaxed"]["min_workday_coverage"]:.0%}): '
        f'{report.pass_relaxed.sum()} CSYT',
        f'- Tập chặt theo tuyến: {strict.tuyen_bv.value_counts(dropna=False).sort_index().to_dict()}',
        f'- Tập chặt theo tỉnh: {strict.ma_tinh.value_counts().sort_index().to_dict()}',
        f'- Chỉ tiêu dùng được trong tập chặt (median tháng ≥ {cfg["fields"]["min_month_median"]}): '
        + ', '.join(f'{f} {int(strict[f"use_{f}"].sum())}' for f in direct),
        f'- Ngày có dung_tuyen + trai_tuyen ≠ bhyt (chỉ báo cáo, không loại): median '
        f'{strict.bhyt_split_mismatch.median():.1%} số ngày/CSYT',
        '', 'Lý do loại của từng CSYT: cột `failed` trong `quality_report.parquet`.',
    ]
    return '\n'.join(lines) + '\n'


def run(snap, cfg):
    raw = clean.load_raw(snap)
    report, last_day, panel = evaluate(raw, pd.read_parquet(snap / 'facilities.parquet'), cfg)
    breaks = clean.etl_breaks(panel, clean.FIELDS[1:], cfg['clean']['etl_break_drop'])
    report.to_parquet(snap / 'quality_report.parquet')
    (snap / 'audit.md').write_text(audit(snap, raw, last_day, report, breaks, cfg), encoding='utf-8')
    print(f'Ngày cuối có dữ liệu {last_day:%d/%m/%Y}; tập chặt {report.pass_strict.sum()} CSYT, '
          f'tập nới {report.pass_relaxed.sum()} CSYT. Đã ghi {snap / "audit.md"}')
