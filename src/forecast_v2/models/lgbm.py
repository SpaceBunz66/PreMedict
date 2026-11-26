"""LightGBM direct multi-horizon (plan §P6): một model cho mọi chuỗi và mọi h, h là một feature.

Mỗi mẫu = (chuỗi, gốc t, h). Feature tại gốc chỉ dùng y tới t: lag, rolling, xu hướng, số ngày từ lần khác 0, khí tượng
quan trắc (w_*, đã trễ 5 ngày). Feature tại ngày đích d = t + h: lịch, lễ, khí hậu, giá trị cùng thứ gần nhất (tới t),
giá trị cùng tet_offset năm trước (ngoài cửa sổ Tết: cùng thứ năm trước, d − 364).
Mọi feature theo y và target chia cho `level` (trung bình 28 ngày có mask = 1 tới gốc), trọng số mẫu = level: loss
tính theo đơn vị gốc, khớp WAPE, mà cây không phải học lại quy mô của từng CSYT. y có mask = 0 coi như thiếu (NaN).
q10/q90: model phân vị riêng (ít vòng hơn q50).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from forecast_v2.calendar_vn import tet_days

LAGS = [0, 1, 2, 3, 4, 5, 6, 13, 20, 27]
WINDOWS = [7, 14, 28, 56]
NOT_TARGET = ['unique_id', 'csyt_id', 'field', 'ds', 'y', 'available_mask']  # còn lại (trừ w_*): covariate tại ngày đích
STATIC = ['field', 'tuyen', 'hang', 'latitude', 'longitude', 'log_scale', 'weekend_open']
# A2 (plan §P8): nhóm feature bỏ được (--drop); lịch cơ bản, lag/rolling và static luôn giữ
HOLIDAY = ['is_holiday', 'is_lam_bu', 'is_day_off', 'holiday_type', 'break_len', 'break_pos', 'days_to_break',
           'days_since_break', 'tet_offset', 'tet_pre', 'tet_post', 'lunar_day', 'lunar_month', 'lunar_leap', 'is_mung1',
           'is_ram', 'school_break', 'regional_holiday']
GROUPS = {'holiday': lambda k: k in HOLIDAY, 'weather': lambda k: k.startswith('w_'),  # w_*: cần job tải ERA5 hằng ngày
          'clim': lambda k: k.startswith('clim_')}


def origin_features(df):
    """Feature của mỗi dòng khi dòng đó là gốc t. df: sắp theo (unique_id, ds), mỗi chuỗi là lịch ngày liên tục."""
    ym = df.y.where(df.available_mask > 0)
    g = ym.groupby(df.unique_id)
    roll = lambda w, how, mp: g.transform(lambda s: getattr(s.rolling(w, min_periods=mp), how)())
    level = roll(28, 'mean', 7).fillna(roll(365, 'mean', 1)).clip(lower=1)  # tháng bị mask cả tháng: lùi về 365 ngày
    f = {'level': level}
    for k in LAGS:
        f[f'lag{k}'] = g.shift(k) / level
    for w in WINDOWS:
        f[f'mean{w}'] = roll(w, 'mean', w // 2) / level
        f[f'median{w}'] = roll(w, 'median', w // 2) / level
        f[f'std{w}'] = roll(w, 'std', w // 2) / level
    f['trend7_28'] = f['mean7'] / f['mean28']
    f['days_since_nz'] = (df.ds - df.ds.where(df.y > 0).groupby(df.unique_id).ffill()).dt.days
    f.update({k: df[k] for k in df if k.startswith('w_')})
    return pd.DataFrame(f).astype('float32')


def last_year_shift(df):
    """Số ngày lùi về 'cùng thời điểm năm trước' của mỗi dòng: trong ±30 ngày quanh Tết thì theo tet_offset, ngoài ra 364."""
    o = df.tet_offset.to_numpy()
    near = np.abs(o) <= 30
    tet = df.ds - pd.to_timedelta(np.where(near, o, 0), unit='D')
    years = tet.dt.year
    prev = pd.Series(tet_days(range(years.min() - 1, years.max())), index=range(years.min(), years.max() + 1))
    gap = (tet - years.map(prev)).dt.days.to_numpy()
    return np.where(near, gap, 364)


def design(df, static, cutoff, horizon, step, drop=(), min_h=1):
    """(X, y/level, level, mẫu train, mẫu dự báo). Gốc train: cutoff − step·k (k ≥ 1) với đích ≤ cutoff, mask đích = 1;
    gốc dự báo: cutoff, h = min_h..horizon (min_h > 1: dự báo xa khi dữ liệu dừng trước ngày cần dự báo).
    drop: tên nhóm trong GROUPS (bỏ holiday thì last_year lùi đúng 364 ngày)."""
    if set(drop) - set(GROUPS):
        raise ValueError(f'Nhóm feature không có: {sorted(set(drop) - set(GROUPS))}; có: {sorted(GROUPS)}')
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    of = origin_features(df)
    off = df.groupby('unique_id').cumcount().to_numpy()  # vị trí trong chuỗi
    ym = df.y.where(df.available_mask > 0).to_numpy('float32')
    lag = (cutoff - df.ds).dt.days.to_numpy()
    # Gốc cần ≥ min_h ngày lịch sử: trước đó là giai đoạn CSYT mới lên hệ thống, mức gần 0 nên y/mức sau 6 tháng
    # lên tới hàng chục, hàng trăm (p90 gốc < 90 ngày: 61; gốc ≥ 169 ngày: 1,6) và model dự báo xa học sai.
    origins = np.flatnonzero((lag >= 0) & (lag % step == 0) & (off >= max(28, min_h)))
    h = np.tile(np.arange(min_h, horizon + 1), len(origins))
    i = np.repeat(origins, horizon - min_h + 1)
    t = i + h  # mỗi chuỗi kéo dài tới cutoff + horizon (features.fold) nên đích luôn cùng chuỗi
    keep = (lag[i] == 0) | ((lag[t] >= 0) & (df.available_mask.to_numpy()[t] > 0))
    i, t, h = i[keep], t[keep], h[keep]
    assert (off[t] == off[i] + h).all(), 'ngày đích nằm ngoài chuỗi: fold phải kéo dài tới cutoff + horizon'

    lv = of.level.to_numpy()[i]
    def back(k):  # y (mask = 1) cách ngày đích k ngày, chia level; k >= h nên luôn ở tại hoặc trước gốc
        j = t - k
        return np.where(off[t] >= k, ym[np.clip(j, 0, None)], np.nan) / lv
    week = 7 * np.ceil(h / 7).astype(int)
    kept = lambda cols: [k for k in cols if not any(GROUPS[g](k) for g in drop)]
    target = kept(k for k in df if k not in NOT_TARGET and not k.startswith('w_'))
    st = static.set_index('unique_id')[STATIC]
    # lat/lon float64 làm LightGBM chuyển cả ma trận sang float64 (gấp đôi RAM); float32 lệch ~0,1 m
    st = st.astype({k: 'float32' for k, d in st.dtypes.items() if d == 'float64'})
    # Thêm từng cột thay pd.concat: concat chép và gộp khối (+4 GB tạm ở 7 triệu dòng × 86 cột), làm backtest
    # snapshot prod hết RAM 15 GB. .array[...] lấy theo vị trí, giữ dtype (category), không căn theo index.
    X = of[kept(of.columns.drop('level'))].iloc[i]
    X.index = pd.RangeIndex(len(X))
    for k in target:
        X[k] = df[k].array[t]
    s = st.index.get_indexer(df.unique_id)[i]
    assert (s >= 0).all(), 'chuỗi thiếu trong static'
    for k in STATIC:
        X[k] = st[k].array[s]
    X['h'] = h.astype('float32')
    X['same_weekday'] = back(week).astype('float32')
    X['last_year'] = back(364 if 'holiday' in drop else last_year_shift(df)[t]).astype('float32')
    for k in X:
        if X[k].dtype == object:
            X[k] = X[k].astype('category')
    y = df.y.to_numpy('float32')[t] / lv
    pred = lag[i] == 0
    return X, y, lv, ~pred, pred, df[['unique_id', 'ds']].iloc[t].reset_index(drop=True)


def _fit(data, objective, alpha, params, rounds, bar=None):
    p = params | {'objective': objective, 'verbosity': -1} | ({'alpha': alpha} if objective == 'quantile' else {})
    return lgb.train(p, data, rounds, callbacks=[lambda env: bar.update()] if bar else None)


def forecast(df, static, cutoff, cfg, save_to=None):
    """save_to: thư mục lưu 3 model (lgbm_q10/q50/q90.txt) cho train-final."""
    c = cfg['models']['lgbm']
    X, y, lv, train, pred, keys = design(df, static, cutoff, cfg['features']['horizon'], c['origin_step'],
                                          cfg['features'].get('drop', ()), cfg['features'].get('min_h', 1))
    Xp = X[pred]
    # một Dataset cho cả 3 model (cùng tham số dataset): chuyển ma trận và chia bin một lần; bỏ X để không giữ hai bản
    # (7 triệu dòng × 86 cột ở gốc 05/2026: các bản sao làm backtest hết RAM 15 GB)
    data = lgb.Dataset(X[train], y[train], weight=lv[train])
    del X
    with tqdm(total=c['rounds'] + 2 * c['interval_rounds'], desc='lgbm q50+q10+q90', unit='vòng', leave=False) as bar:
        models = {'q50': _fit(data, c['objective'], 0.5, c['params'], c['rounds'], bar)} | {
            f'q{round(a * 100)}': _fit(data, 'quantile', a, c['params'], c['interval_rounds'], bar) for a in (0.1, 0.9)}
    q = np.column_stack([models[k].predict(Xp) for k in ('q10', 'q50', 'q90')])
    q = np.clip(q, 0, None) * lv[pred, None]
    s = np.sort(q, axis=1)
    # q50 giữ của tweedie: ngày lễ q50 hay thấp hơn q10 (36% điểm lễ backtest), lấy trung vị 3 dự báo đẩy q50 lên mức
    # ngày thường (WAPE lễ validation 0,763 -> 0,893). Khoảng = khoảng cách q10/q90 tới trung vị sau sắp xếp, dời về q50
    # (không chéo thì đúng bằng q10/q90 của model). Kẹp min/max thì phía dưới rộng 0, hệ số A5 thành inf.
    out = keys[pred].reset_index(drop=True).assign(q10=np.clip(q[:, 1] - (s[:, 1] - s[:, 0]), 0, None), q50=q[:, 1],
                                                    q90=q[:, 1] + (s[:, 2] - s[:, 1]))
    out.attrs['importance'] = pd.Series(models['q50'].feature_importance('gain'), index=Xp.columns)
    out.attrs['train_rows'] = int(train.sum())
    if save_to:
        for k, m in models.items():
            m.save_model(save_to / f'lgbm_{k}.txt')
    return out
