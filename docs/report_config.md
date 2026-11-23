# Tóm tắt cấu hình cho báo cáo

Nguồn cho phần "Thiết lập thí nghiệm" của báo cáo (A7). Số liệu lấy từ `configs/base.yaml`, `config.yaml` của từng run và `metrics.md` trong `runs/`. Cập nhật khi có cấu hình hay run mới.

## 1. Dữ liệu

- Snapshot 2026-11-18. Dữ liệu ngày từ 01/01/2021 tới **07/06/2026** (ETL trong DB dừng ở ngày này).
- Tập chặt: **131 CSYT, 318 chuỗi**. 5 chỉ tiêu dự báo trực tiếp: `tong`, `luot_kham`, `ngoai_tru`, `noi_tru`, `cap_cuu`.
- Horizon 30 ngày. Bảng feature có 58 cột: lịch, lễ/Tết/âm lịch, 24 cột khí tượng quan trắc `w_*`, 3 cột khí hậu `clim_*`.

## 2. Chia train / validation / test

Backtest theo thời gian với cửa sổ train mở rộng (rolling-origin), không chia ngẫu nhiên. Tại mỗi gốc c: train lại từ đầu trên dữ liệu 01/01/2021 → c, dự báo c+1 → c+30, chấm điểm trên 30 ngày đó.

| Tập | Gốc (ngày train cuối) | Khoảng chấm điểm | Ghi chú |
|---|---|---|---|
| validation | 15/01/2025 | 16/01 → 14/02/2025 | bao trọn Tết 2025 (29/01) |
| validation | 30/09/2025 | 01/10 → 30/10/2025 | |
| validation | 30/11/2025 | 01/12 → 30/12/2025 | |
| test | 01/01/2026 | 02/01 → 31/01/2026 | |
| test | 31/01/2026 | 01/02 → 02/03/2026 | bao trọn Tết 2026 (17/02) |
| test | 02/03/2026 | 03/03 → 01/04/2026 | |
| test | 01/04/2026 | 02/04 → 01/05/2026 | |
| test | 01/05/2026 | 02/05 → 31/05/2026 | |

- **Validation** dùng cho mọi lựa chọn: tune Optuna, ablation, cấu hình TiDE (A3), cổng giữ/bỏ TiDE (A4), hệ số khoảng tin cậy (A5).
- **Test** chỉ để báo cáo. Khoảng test bắt đầu sau khi khoảng validation cuối kết thúc (30/12/2025).
- Chống leakage tại mỗi gốc (`features.fold`):
  - mask ngày hợp lệ tính lại chỉ từ dữ liệu tới c;
  - xóa khí tượng quan trắc `w_*` sau c (tương lai chỉ còn lịch, lễ/Tết, khí hậu);
  - `log_scale`, `weekend_open` và mẫu số MASE lấy từ 365 ngày tới c.
- Khoảng train thực tế theo model:
  - LightGBM: gốc train c−2, c−4, … với mọi ngày đích ≤ c;
  - TiDE: giữ 30 ngày cuối (c−29 → c) cho early stopping, nên trọng số học từ dữ liệu tới khoảng c−30.
- **Hạn chế:** tập chặt lọc theo chất lượng 365 ngày cuối snapshot (08/06/2025 → 07/06/2026, `quality.py`), trùng 2 gốc validation cuối và cả 5 gốc test. So sánh giữa các model (kể cả Prophet) vẫn công bằng vì cùng tập CSYT, nhưng WAPE tuyệt đối có thể hơi lạc quan.

## 3. Cấu hình model

**LightGBM** (direct multi-horizon, `h` là feature): objective tweedie, 600 vòng (200 vòng cho model khoảng tin cậy), learning_rate 0,05, num_leaves 63, min_data_in_leaf 200, feature_fraction 0,8, bagging_fraction 0,7 (freq 1), lambda_l2 1,0, max_bin 127, seed 0, gốc train cách 2 ngày. Tune A1 (Optuna, 10 trial tweedie) không hơn config ≥ 0,5 điểm nên giữ config.

**TiDE** (neuralforecast 3.3.0, CPU): input_size 56, hidden_size 512 (mặc định), MQLoss [0,1; 0,5; 0,9], q50 = median, tự scale theo mức chuỗi (`scaler_type='identity'`), batch_size 64 chuỗi, windows_batch_size 512, learning_rate 0,001 không giảm (num_lr_decays −1), early stopping kiểm tra mỗi 50 step, patience 5, seed 0. Mặc định max_steps 1000.

## 4. Thử cấu hình TiDE (A3, trên 3 gốc validation)

Người dùng chạy trên laptop ngày 21/11/2026. Chỉ đổi các tham số trong cột "Thay đổi"; còn lại như mục 3.

| Cấu hình | Thay đổi | Run | Giây/gốc | WAPE 15/01/25 | 30/09/25 | 30/11/25 | **WAPE val** |
|---|---|---|---|---|---|---|---|
| s1000 | mặc định | `20261120-213929_mean4-prophet_prod-tide` | 1.642 | 0,474 | 0,245 | 0,242 | 0,304 |
| **s2000** | max_steps 2000 | `20261121-142639_tide_a3-s2000` | 2.606 | 0,474 | 0,229 | 0,242 | **0,299** |
| s3000 | max_steps 3000 | `20261121-164635_tide_a3-s3000` | 2.794 | 0,474 | 0,242 | 0,242 | 0,304 |
| s3000-h256 | max_steps 3000, hidden_size 256 | `20261121-185718_tide_a3-s3000-h256` | 2.609 | 0,477 | 0,255 | 0,248 | 0,311 |
| *lgbm (so sánh)* | config mục 3 | `20261119-114722_...` | | 0,304 | 0,219 | 0,263 | *0,2585* |

- Step dừng thật (early stopping), theo gốc 15/01/25 · 30/09/25 · 30/11/25: s2000 800 · 2000 · 2000; s3000 800 · 2400 · 2000; s3000-h256 1000 · 2050 · 2350.
- Ở gốc Tết 15/01/25, cả s1000, s2000 và s3000 đều dừng ở step 800 nên dự báo giống hệt nhau. Cửa sổ early stopping (16/12/2024 → 15/01/2025) không có Tết.
- Chênh s2000/s3000 chỉ ở gốc 30/09 (0,229 so với 0,242): mức nhiễu train.
- Chạy lại cùng cấu hình (run `-211306`, `-231955`) cho dự báo giống từng byte: TiDE seed 0 tái lập được. Hai run này đã xóa sau khi đối chiếu (23/11/2026).
- Khoảng cách với LightGBM nằm ở Tết. Ngoài Tết ±21 ngày: tide 0,236–0,251, lgbm 0,242. Trong Tết ±21: tide 0,474, lgbm 0,304. Ngày lễ: 6,76 so với 0,89.
- Cổng A4 (validation):
  - (a) WAPE ≤ lgbm: 0,299 > 0,2585, không đạt;
  - (b) trộn với lgbm giảm ≥ 0,5 điểm: tốt nhất 0,2577 (20% tide), chỉ −0,08 điểm, không đạt;
  - (c) thắng rõ một chỉ tiêu: `cap_cuu` lgbm 0,344, tide 0,365–0,384, không đạt.

## 5. TiDE trên test (chỉ để báo cáo)

Run `20261122-084951_tide_a3-s2000-test`: cấu hình s2000, 5 gốc test, 3.200 giây/gốc (tổng 4 giờ 27 phút). TiDE đã bị loại ở cổng A4 trên validation; số test dưới đây không dùng để chọn lại. lgbm lấy từ `20261119-114722_...`. cover80 của cả hai là trước hiệu chỉnh A5.

| Nhóm (test) | lgbm | tide |
|---|---|---|
| **WAPE tổng** | **0,256** | 0,273 |
| Ngoài Tết ±21 | 0,237 | 0,256 |
| Trong Tết ±21 | 0,324 | 0,334 |
| Ngày lễ | 0,861 | 2,208 |
| h 1–7 · 8–14 · 15–30 | 0,235 · 0,245 · 0,277 | 0,255 · 0,262 · 0,292 |
| `cap_cuu` | 0,957 | 0,733 |
| `ngoai_tru` | 0,399 | 0,378 |
| `noi_tru` | 0,515 | 0,535 |
| Bắc Ninh (24) | 0,284 | 0,251 |
| Tổng tỉnh × chỉ tiêu | 0,165 | 0,193 |
| cover80 | 0,681 | 0,650 |

- Test xác nhận validation: lgbm tốt hơn tổng thể, ở mọi nhóm horizon và quanh Tết. TiDE vẫn yếu ở ngày lễ.
- Trên test TiDE thắng ở `cap_cuu`, `ngoai_tru` và Bắc Ninh. Ở validation lgbm thắng `cap_cuu` (cổng A4c). Không đổi quyết định A4 vì dùng test để chọn là rò rỉ; ghi vào hạn chế và hướng tiếp (chuỗi nhỏ `cap_cuu` nhiễu, n = 1.145).

## 6. Test lần cuối và train-final (A7, 23/11/2026)

- **Test lần cuối** của LightGBM là run `20261119-114722_...`: 5 gốc test chạy một lần với đúng config mục 3. Sau run này config không đổi (A1 giữ config, A2 chưa chốt bỏ nhóm nào); code lgbm chỉ thêm thanh tiến độ và cờ `--drop` (mặc định không bỏ gì), không đổi kết quả. Vì vậy không chạy lại.
- **Train-final** `artifacts/lgbm/20261123-093332` (commit `3451bf2`, git_dirty false):
  - config mục 3, đủ bộ feature A3 (có `w_*` và `clim_*`);
  - train trên toàn bộ dữ liệu tới 07/06/2026: 2.826.526 mẫu, 363 giây;
  - dự báo 08/06 → 07/07/2026 cho 318 chuỗi (9.540 dòng `forecast.csv`, đủ q10/q50/q90);
  - khoảng tin cậy hiệu chỉnh A5 bằng `interval_factors_lgbm.csv` của run `20261119-114722_...` (fit trên 3 gốc validation). Độ rộng (q90 − q10) trung bình tăng từ 0,59 lên 0,67 lần q50.
- q50 giống hệt bản train-final 20/11 (`20261120-114258`, làm trước A5): model không đổi, chỉ khoảng tin cậy khác. Bản 20/11 đã xóa sau khi đối chiếu (23/11/2026).
- Nếu chốt bỏ ERA5 (bộ feature A1) thì phải train-final lại. Lệnh `train-final` hiện chưa có cờ `--drop`, và hệ số A5 cũng phải lấy từ run A1.

## 7. Dự báo xa cho demo (train-far, 23/11/2026)

Người dùng cần dashboard dự báo từ hôm nay (23/11/2026). Query read-only ngày 23/11/2026 xác nhận DB vẫn dừng: khoảng 2.000 CSYT báo cáo tới 05/06/2026, từ 08/06 chỉ còn 2–17 CSYT, ngày cuối có số là 18/06/2026. Vì vậy dự báo 23/11 → 22/12/2026 được tính thẳng từ ngày 07/06/2026, cách 168 ngày không có số liệu.

- **Lệnh:** `train-far --start 2026-11-23` → `artifacts/lgbm_far/20261123-110548`.
- **Model:** LightGBM như mục 3, đủ bộ feature A3, nhưng train với h = 169–198 thay cho 1–30 (`lgbm.design(min_h=...)`). Bảng feature được dựng lại trong bộ nhớ, kéo dài tới 22/12/2026; không ghi đè `features.parquet`.
- **Gốc train phải có ≥ min_h ngày lịch sử.** Trước mốc đó là giai đoạn CSYT mới lên hệ thống: mức gần 0 nên tỉ lệ y sau 6 tháng / mức tại gốc cực lớn (p90 là 61 với gốc < 90 ngày, 13 với gốc 90–168 ngày, 1,6 với gốc ≥ 169 ngày). Khi chưa có quy tắc này, backtest cho WAPE 0,975 và bias +0,76.
- **Thay bằng cùng kỳ năm trước:** chuỗi có dưới 14 ngày mask = 1 trong 28 ngày trước gốc thì q50 lấy số cùng kỳ năm trước (lùi 364 ngày); q10/q90 giữ tỉ lệ với q50 của model. Lần này áp cho 27 chuỗi `ngoai_tru`, do ETL gãy 02–05/2026: model cho các chuỗi này gấp 4–12 lần cùng kỳ (Hà Nội 01K61, 01I14, 01037…).
- **Backtest cùng mùa năm trước:** dữ liệu tới 08/06/2025, dự báo 24/11 → 23/12/2025 (lùi 364 ngày nên trùng thứ), 197 chuỗi, 5.791 điểm.

  | | WAPE | bias | cover80 |
  |---|---|---|---|
  | lgbm_far | **0,312** | +0,030 | 0,741 |
  | cùng kỳ năm trước | 0,333 | +0,059 | – |

  Để so sánh, dự báo 30 ngày thường (test) có WAPE 0,256.
- **Bản cuối:** train trên các gốc tới 20/12/2025, 1.493.325 mẫu, 155 giây. Tỉ lệ q50 / cùng kỳ 2025 theo chỉ tiêu: `tong` 0,92, `luot_kham` 0,95, `noi_tru` 0,95, `cap_cuu` 1,00, `ngoai_tru` 1,00 (cùng kỳ năm trước).
- **Hạn chế:**
  - chỉ một lần backtest;
  - model backtest học từ ít dữ liệu hơn bản cuối (gốc tới 21/12/2024), nên số đo có thể hơi bi quan;
  - q10/q90 chưa hiệu chỉnh, vì hệ số A5 chỉ fit cho h ≤ 30.
- **Dashboard** (`dashboard/build_data.py`) mặc định đọc artifact `lgbm_far` mới nhất. Trên trang có dòng ghi chú và vạch "bỏ qua 168 ngày" nói rõ đây là dự báo xa.
