"""P7: metric trên các điểm mask = 1 (plan §P7)."""
import warnings

import numpy as np
import pandas as pd

QUANTILES = {'q10': 0.1, 'q50': 0.5, 'q90': 0.9}
H_BINS, H_LABELS = [0, 7, 14, 30], ['1–7', '8–14', '15–30']


def mase_scale(df, cutoff):
    """Mẫu số MASE theo chuỗi: MAE của seasonal-naive-7 trong 365 ngày tới cutoff, chỉ trên cặp ngày đều có mask = 1.
    df: dữ liệu fold, mỗi chuỗi là lịch ngày liên tục, tăng dần."""
    p = df[(df.ds <= cutoff) & (df.ds > cutoff - pd.Timedelta(days=365))]
    g = p.groupby('unique_id')
    d = (p.y - g.y.shift(7)).abs().where((p.available_mask > 0) & (g.available_mask.shift(7) > 0))
    return d.groupby(p.unique_id).mean()


def score(e, by=()):
    """e: dòng đánh giá đã lọc mask = 1, cột model, set, y, q10, q50, q90, scale. by: tên cột hoặc Series.

    wape, bias, pinball (trung bình 3 phân vị) chia cho tổng y; mase = trung bình |sai số| / scale; cover80 = tỉ lệ y
    trong [q10, q90]. Model không có khoảng tin cậy: pinball, cover80 NaN.
    """
    pin = sum(np.maximum(t * (e.y - e[q]), (t - 1) * (e.y - e[q])) for q, t in QUANTILES.items()) / len(QUANTILES)
    e = e.assign(ae=(e.q50 - e.y).abs(), err=e.q50 - e.y, pin=pin,
                 cover=((e.y >= e.q10) & (e.y <= e.q90)).astype(float).where(e.q10.notna()),
                 sae=(e.q50 - e.y).abs() / e.scale.where(e.scale > 0))
    g = e.groupby(['model', 'set', *by], observed=True)
    s = g[['y', 'ae', 'err']].sum()
    return pd.DataFrame({'n': g.size(), 'wape': s.ae / s.y, 'mase': g.sae.mean(), 'bias': s.err / s.y,
                         'pinball': g.pin.sum(min_count=1) / s.y, 'cover80': g.cover.mean()})


def report(e):
    """Các lát cắt của plan §P7. Tổng tỉnh = cộng dồn dự báo CSYT (bottom-up) rồi mới tính sai số.

    shortcut: chưa có lát ngày thời tiết cực đoan và chỉ tiêu suy ra; thêm ở P8. Wilcoxon: hàm versus.
    """
    prov = e.groupby(['model', 'set', 'ma_tinh', 'field', 'ds'], observed=True)[['y', 'q50']].sum().reset_index()
    prov = prov.assign(q10=np.nan, q90=np.nan, scale=np.nan)
    out = {
        'tổng': score(e),
        'horizon': score(e, [pd.cut(e.h, H_BINS, labels=H_LABELS).rename('h')]),
        'chỉ tiêu': score(e, ['field']),
        'tuyến': score(e, ['tuyen']),
        'Tết ±21 ngày': score(e, [e.tet_offset.abs().le(21).rename('tet')]),
        'ngày lễ': score(e, [e.is_holiday.gt(0).rename('holiday')]),
        'Bắc Ninh (24)': score(e[e.ma_tinh == '24']),
        'tổng tỉnh × chỉ tiêu': score(prov),
    }
    rows = []
    for name, s in out.items():
        s = s.reset_index()
        by = [c for c in s.columns if c not in ('model', 'set', 'n', 'wape', 'mase', 'bias', 'pinball', 'cover80')]
        rows.append(s.assign(slice=name, group=s[by[0]].astype(str) if by else '').drop(columns=by))
    with warnings.catch_warnings():  # cột toàn NaN (model không có khoảng tin cậy): kiểu kết quả vẫn là float
        warnings.simplefilter('ignore', FutureWarning)
        return pd.concat(rows, ignore_index=True).set_index(['slice', 'group', 'set', 'model'])


def versus(e, ref='prophet_prod'):
    """Kiểm định Wilcoxon ghép cặp theo chuỗi (plan §P7): tổng |sai số| của mỗi chuỗi, model so với ref, cùng các điểm.
    better_share: tỉ lệ chuỗi model có sai số nhỏ hơn ref."""
    from scipy.stats import wilcoxon
    err = (e.q50 - e.y).abs().groupby([e.set, e.unique_id, e.model]).sum().unstack('model')
    rows = []
    for kind, d in err.groupby(level='set'):
        for m in d.columns.drop(ref):
            x = d[[m, ref]].dropna()
            rows.append({'set': kind, 'model': m, 'series': len(x), 'better_share': (x[m] < x[ref]).mean(),
                         'p_value': wilcoxon(x[m], x[ref]).pvalue})
    return pd.DataFrame(rows).set_index(['set', 'model'])


def interval_factors(e, miss=0.1):
    """A5 (plan §P8): conformal theo nhóm h. e: dòng mask = 1 của các gốc validation, cột model, h, y, q10, q50, q90.

    Giãn/co mỗi đầu khoảng quanh q50 theo một hệ số: q10' = q50 − lo·(q50 − q10), q90' = q50 + hi·(q90 − q50), với lo
    (hi) = phân vị 1 − miss của (q50 − y)/(q50 − q10) ((y − q50)/(q90 − q50)), nên mỗi đầu sai đúng tỉ lệ miss trên e.
    Tỉ lệ theo độ rộng của chính model nên dùng chung được cho chuỗi lớn và nhỏ. Trả về bảng index (model, h) cột lo, hi.
    """
    e = e[e.q10.notna()]

    def ratio(over, width):  # độ rộng 0: y vượt q50 thì không hệ số nào phủ được (inf), không thì luôn phủ (-inf)
        return np.where(width > 0, over / np.where(width > 0, width, 1), np.where(over > 0, np.inf, -np.inf))

    r = pd.DataFrame({'lo': ratio(e.q50 - e.y, e.q50 - e.q10), 'hi': ratio(e.y - e.q50, e.q90 - e.q50)}, index=e.index)
    g = r.groupby([e.model, pd.cut(e.h, H_BINS, labels=H_LABELS).rename('h')], observed=True)
    return g.quantile(1 - miss, interpolation='higher').clip(lower=0)


def calibrate(e, f):
    """Áp hệ số của interval_factors vào q10/q90 (q50 giữ nguyên); model không có trong f giữ khoảng cũ."""
    k = pd.MultiIndex.from_arrays([e.model, pd.cut(e.h, H_BINS, labels=H_LABELS).astype(str)])
    f = f.set_axis(f.index.set_levels(f.index.levels[1].astype(str), level=1)).reindex(k)
    lo, hi = f.lo.to_numpy(), f.hi.to_numpy()
    return e.assign(q10=np.where(np.isnan(lo), e.q10, np.clip(e.q50 - lo * (e.q50 - e.q10), 0, None)),
                    q90=np.where(np.isnan(hi), e.q90, e.q50 + hi * (e.q90 - e.q50)))
