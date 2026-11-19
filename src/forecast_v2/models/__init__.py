"""Models (plan §P6). Mỗi model là một hàm forecast(df, static, cutoff, cfg) nhận dữ liệu của một fold
(features.fold) và trả về DataFrame [unique_id, ds, q10, q50, q90] cho cutoff+1..cutoff+horizon; q10/q90 NaN nếu
model không có khoảng tin cậy. Model chỉ được đọc y và w_* của các dòng ds <= cutoff.
"""
