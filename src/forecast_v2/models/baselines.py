"""Baseline naive (plan §P6): seasonal-naive-7 và trung bình 4 ngày cùng thứ gần nhất. Không có khoảng tin cậy."""
import numpy as np
import pandas as pd


def _by_weekday(df, cutoff, n, masked):
    """Dự báo = trung bình y của n ngày cùng thứ gần nhất tới cutoff, theo chuỗi; masked: chỉ lấy ngày có mask = 1."""
    hist = df[df.ds <= cutoff]
    if masked:
        hist = hist[hist.available_mask > 0]
    hist = hist.sort_values('ds')
    hist = hist.groupby(['unique_id', hist.ds.dt.dayofweek]).tail(n)
    v = hist.groupby(['unique_id', hist.ds.dt.dayofweek]).y.mean()
    fut = df.loc[df.ds > cutoff, ['unique_id', 'ds']]
    q50 = v.reindex(pd.MultiIndex.from_arrays([fut.unique_id, fut.ds.dt.dayofweek])).to_numpy()
    return fut.assign(q10=np.nan, q50=q50, q90=np.nan).reset_index(drop=True)


def snaive7(df, static, cutoff, cfg):
    """Lặp lại 7 ngày cuối tới cutoff, kể cả ngày mask = 0 (đúng định nghĩa, cũng là mẫu số MASE)."""
    return _by_weekday(df, cutoff, 1, masked=False)


def mean4(df, static, cutoff, cfg):
    """Trung bình 4 ngày cùng thứ gần nhất có mask = 1 (tháng bị mask cả tháng thì lùi xa hơn)."""
    return _by_weekday(df, cutoff, 4, masked=True)
