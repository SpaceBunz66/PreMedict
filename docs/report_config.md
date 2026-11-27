# Tóm tắt cấu hình cho báo cáo

Nguồn cho phần "Thiết lập thí nghiệm" của báo cáo (A7). Số liệu lấy từ `configs/base.yaml`, `config.yaml` của từng run và `metrics.md` trong `runs/`. Cập nhật khi có cấu hình hay run mới.

## 1. Dữ liệu

- Snapshot 2026-11-18. Dữ liệu ngày từ 01/01/2021 tới **07/06/2026** (ETL trong DB dừng ở ngày này).
  - Kiểm tra lại ngày 25/11/2026 (read-only): cả 35 partition (theo `ma_tinh`) của `hoc.dtm_hoc_kcb_theo_ngay` dừng trong khoảng 30/05–18/06/2026, sau đó chỉ còn dòng 0 đệm tới 31/12/2026. Từ lần DB khởi động 24/09/2026 tới nay có 0 lượt insert/update/delete vào các partition này, dù các bảng khác trong DB vẫn được ghi.
  - Bảng nguồn `hoc.kcb_tong_hop` (`tong` của datamart = số hồ sơ theo ngày ra; tổng tháng 04–05/2026 khớp gần hết) cũng gần như dừng: sau 07/06/2026 chỉ có ~25 nghìn hồ sơ của 12 CSYT tỉnh 34 (01–19/07/2026) và vài hồ sơ lẻ. Trong tập model chỉ 20 CSYT có số liệu, và chỉ cho một ngày 08/06/2026 (chưa đủ ngày). Không có dữ liệu mới đủ để đánh giá.
  - Quy ước: dữ liệu mới về sau (nếu có) chỉ dùng để đánh giá dự báo đã train tới 07/06/2026, không đưa vào train.
- Định nghĩa target (hàm ETL `hoc.ins_datamart_kcb` trong DB): `time_id` = **ngày ra** (`ngay_ra`, YYYYMMDD) của hồ sơ trong `kcb_tong_hop`, đã bỏ trùng theo (`ma_cskcb`, `ma_lk`). `tong` = `luot_ra` = số hồ sơ ra trong ngày; `luot_kham`/`ngoai_tru`/`noi_tru` theo `ma_loai_kcb` (1 / 2,5,8 / 3,4,9); `cap_cuu` = `ma_doituong_kcb` '2'. Hàm tạo dòng 0 cho mọi CSYT × mọi ngày của khoảng được chạy, nên số 0 không có nghĩa là không có lượt khám.
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
| *lgbm (so sánh)* | config mục 3 | `20261123-161154_...` | | 0,300 | 0,221 | 0,264 | *0,2582* |

- Step dừng thật (early stopping), theo gốc 15/01/25 · 30/09/25 · 30/11/25: s2000 800 · 2000 · 2000; s3000 800 · 2400 · 2000; s3000-h256 1000 · 2050 · 2350.
- Ở gốc Tết 15/01/25, cả s1000, s2000 và s3000 đều dừng ở step 800 nên dự báo giống hệt nhau. Cửa sổ early stopping (16/12/2024 → 15/01/2025) không có Tết.
- Chênh s2000/s3000 chỉ ở gốc 30/09 (0,229 so với 0,242): mức nhiễu train.
- Chạy lại cùng cấu hình (run `-211306`, `-231955`) cho dự báo giống từng byte: TiDE seed 0 tái lập được. Hai run này đã xóa sau khi đối chiếu (23/11/2026).
- Khoảng cách với LightGBM nằm ở Tết. Ngoài Tết ±21 ngày: tide 0,236–0,251, lgbm 0,243. Trong Tết ±21: tide 0,474, lgbm 0,300. Ngày lễ: 6,76 so với 0,76.
- Cổng A4 (validation; số lgbm đã cập nhật theo bản sửa q50 ở mục 6, kết luận không đổi):
  - (a) WAPE ≤ lgbm: 0,299 > 0,2582, không đạt;
  - (b) trộn với lgbm giảm ≥ 0,5 điểm: tốt nhất 0,2567 (10% tide), chỉ −0,15 điểm, không đạt;
  - (c) thắng rõ một chỉ tiêu: `cap_cuu` lgbm 0,344, tide 0,365–0,384, không đạt.

## 5. TiDE trên test (chỉ để báo cáo)

Run `20261122-084951_tide_a3-s2000-test`: cấu hình s2000, 5 gốc test, 3.200 giây/gốc (tổng 4 giờ 27 phút). TiDE đã bị loại ở cổng A4 trên validation; số test dưới đây không dùng để chọn lại. lgbm lấy từ `20261123-161154_...` (bản sửa q50, mục 6). cover80 của cả hai là trước hiệu chỉnh A5.

| Nhóm (test) | lgbm | tide |
|---|---|---|
| **WAPE tổng** | **0,256** | 0,273 |
| Ngoài Tết ±21 | 0,238 | 0,256 |
| Trong Tết ±21 | 0,322 | 0,334 |
| Ngày lễ | 0,829 | 2,208 |
| h 1–7 · 8–14 · 15–30 | 0,234 · 0,245 · 0,277 | 0,255 · 0,262 · 0,292 |
| `cap_cuu` | 0,927 | 0,733 |
| `ngoai_tru` | 0,404 | 0,378 |
| `noi_tru` | 0,523 | 0,535 |
| Bắc Ninh (24) | 0,285 | 0,251 |
| Tổng tỉnh × chỉ tiêu | 0,164 | 0,193 |
| cover80 | 0,758 | 0,650 |

- Test xác nhận validation: lgbm tốt hơn tổng thể, ở mọi nhóm horizon và quanh Tết. TiDE vẫn yếu ở ngày lễ.
- Trên test TiDE thắng ở `cap_cuu`, `ngoai_tru` và Bắc Ninh. Ở validation lgbm thắng `cap_cuu` (cổng A4c). Không đổi quyết định A4 vì dùng test để chọn là rò rỉ; ghi vào hạn chế và hướng tiếp (chuỗi nhỏ `cap_cuu` nhiễu, n = 1.145).

## 6. Test lần cuối và train-final (A7, 23/11/2026)

- **Test lần cuối** của LightGBM là run `20261123-161154_...`: đủ 5 model, 8 gốc, config mục 3. Run 19/11 (`20261119-114722_...`) đã xóa vì chạy với lỗi phân vị dưới đây; ngoài lgbm, mọi model trong run mới giống hệt run 19/11.
- **Sửa q50 của LightGBM (23/11/2026, commit `146f590`, `689309f`).**
  - Lỗi: code cũ sắp xếp 3 dự báo q10/q50/q90 rồi lấy giá trị giữa làm q50. Khi q50 của tweedie thấp hơn model phân vị (12% điểm backtest, 36% ngày lễ), q50 bị thay bằng số của model q10/q90, thường là mức ngày thường.
  - Sửa: giữ q50 của tweedie. Khoảng giữ khoảng cách q10/q90 tới trung vị sau sắp xếp, dời về q50; không chéo thì đúng q10/q90 của model. Cách kẹp min/max (thử trước) làm phía dưới rộng 0 ở khoảng 10% điểm, hệ số A5 thành inf.
  - Quyết định theo validation: ngày lễ 0,893 → 0,763, Tết ±21 0,304 → 0,300, tổng 0,2585 → 0,2582. Test: tổng 0,2564 → 0,2562, ngày lễ 0,861 → 0,829.
  - Không chạy lại: 4 run ablation A2 (bản sửa đổi WAPE tổng 0,02 điểm, kết luận không đổi) và Optuna A1 (`tune.py` chỉ train model q50, vốn không sắp xếp).
- **Train-final** `artifacts/lgbm/20261123-165545` (commit `fad98b8`; git_dirty true chỉ vì `dashboard/` và `docs/` đang sửa dở, `src/` và `configs/` sạch):
  - config mục 3, đủ bộ feature A3 (có `w_*` và `clim_*`);
  - train trên toàn bộ dữ liệu tới 07/06/2026: 2.826.526 mẫu, 370 giây;
  - dự báo 08/06 → 07/07/2026 cho 318 chuỗi (9.540 dòng `forecast.csv`, đủ q10/q50/q90);
  - khoảng tin cậy hiệu chỉnh A5 bằng `interval_factors_lgbm.csv` của run `20261123-161154_...` (fit trên 3 gốc validation). Hệ số lo/hi theo h 1–7 · 8–14 · 15–30: 1,03/0,97 · 1,13/1,29 · 1,25/1,09. cover80 validation 0,75 → 0,80, test 0,76 → 0,82. Độ rộng (q90 − q10) trung bình 0,67 lần q50.
- So với bản train-final trước bản sửa (`20261123-093332`, đã xóa): q50 đổi ở 1,5% số dòng, vì khoảng 08/06 → 07/07 hầu như không có ngày lễ; tổng q50 +0,04%.
- Nếu chốt bỏ ERA5 (bộ feature A1) thì phải train-final lại. Lệnh `train-final` hiện chưa có cờ `--drop`, và hệ số A5 cũng phải lấy từ run A1.

## 7. Dự báo xa cho demo (train-far, 23/11/2026)

Người dùng cần dashboard dự báo từ hôm nay (23/11/2026). Query read-only ngày 23/11/2026 xác nhận DB vẫn dừng: khoảng 2.000 CSYT báo cáo tới 05/06/2026, từ 08/06 chỉ còn 2–17 CSYT, ngày cuối có số là 18/06/2026. Vì vậy dự báo 23/11 → 22/12/2026 được tính thẳng từ ngày 07/06/2026, cách 168 ngày không có số liệu.

- **Lệnh:** `train-far --start 2026-11-23` → `artifacts/lgbm_far/20261123-170211` (sau bản sửa q50, mục 6). Bản đầu `20261123-110548` (trước bản sửa) giữ lại vì số chỉ tiêu suy ra bên dưới lấy từ bản đó.
- **Model:** LightGBM như mục 3, đủ bộ feature A3, nhưng train với h = 169–198 thay cho 1–30 (`lgbm.design(min_h=...)`). Bảng feature được dựng lại trong bộ nhớ, kéo dài tới 22/12/2026; không ghi đè `features.parquet`.
- **Gốc train phải có ≥ min_h ngày lịch sử.** Trước mốc đó là giai đoạn CSYT mới lên hệ thống: mức gần 0 nên tỉ lệ y sau 6 tháng / mức tại gốc cực lớn (p90 là 61 với gốc < 90 ngày, 13 với gốc 90–168 ngày, 1,6 với gốc ≥ 169 ngày). Khi chưa có quy tắc này, backtest cho WAPE 0,975 và bias +0,76.
- **Thay bằng cùng kỳ năm trước:** chuỗi có dưới 14 ngày mask = 1 trong 28 ngày trước gốc thì q50 lấy số cùng kỳ năm trước (lùi 364 ngày); q10/q90 giữ tỉ lệ với q50 của model. Lần này áp cho 27 chuỗi `ngoai_tru`, do ETL gãy 02–05/2026: model cho các chuỗi này gấp 4–12 lần cùng kỳ (Hà Nội 01K61, 01I14, 01037…).
- **Backtest cùng mùa năm trước:** dữ liệu tới 08/06/2025, dự báo 24/11 → 23/12/2025 (lùi 364 ngày nên trùng thứ), 197 chuỗi, 5.791 điểm.

  | | WAPE | bias | cover80 |
  |---|---|---|---|
  | lgbm_far | **0,315** | +0,031 | 0,760 |
  | cùng kỳ năm trước | 0,333 | +0,059 | – |

  Để so sánh, dự báo 30 ngày thường (test) có WAPE 0,256. Bản trước sửa q50: 0,312, +0,030, cover80 0,741.
- **Bản cuối:** train trên các gốc tới 20/12/2025, 1.493.325 mẫu, 173 giây. Tỉ lệ q50 / cùng kỳ 2025 theo chỉ tiêu: `tong` 0,93, `luot_kham` 0,96, `noi_tru` 0,95, `cap_cuu` 1,00, `ngoai_tru` 1,00 (cùng kỳ năm trước). So với bản 110548: q50 đổi ở 4,5% số dòng, tổng `tong` 30 ngày 188.192 so với 186.188.
- **Hạn chế:**
  - chỉ một lần backtest;
  - model backtest học từ ít dữ liệu hơn bản cuối (gốc tới 21/12/2024), nên số đo có thể hơi bi quan;
  - q10/q90 chưa hiệu chỉnh, vì hệ số A5 chỉ fit cho h ≤ 30.
- **Dashboard** (`dashboard/build_data.py`) mặc định đọc artifact `lgbm_far` mới nhất. Trên trang có dòng ghi chú và vạch "bỏ qua 168 ngày" nói rõ đây là dự báo xa.
  - Chỉ tiêu suy ra (chỉ trên dashboard, không train): `bhyt`, `dung_tuyen`, `trai_tuyen` = dự báo `tong` (q10/q50/q90) × tỉ lệ field/`tong` trong 28 ngày có số liệu cuối của từng CSYT. Backtest tỉ lệ (gốc 08/06/2025, chấm 24/11 → 23/12/2025, 131 CSYT, giả định `tong` đúng): WAPE do tỉ lệ `bhyt` 0, `dung_tuyen` 0,071, `trai_tuyen` 1,087. Trái tuyến từng CSYT kém tin cậy vì chỉ 16–30 CSYT có ghi và bật/tắt thất thường; cộng toàn bộ thì tỉ lệ trái tuyến/BHYT ổn định ở 0,023–0,048 mỗi tháng từ 10/2024. Bản 20261123-110548: 30 ngày q50 `tong` 186.183, `dung_tuyen` 156.935, `trai_tuyen` 4.768.
  - Tai nạn thương tích (`tn_*`) chưa dự báo: chỉ chiếm 1,6% `tong`, 11% số ngày có ca.

## 8. DB production (kiểm tra 25/11/2026)

Đồng nghiệp đưa cấu hình DB của bản deploy k8s (`hoc-ai:1.4.5`): DB prod (nay là `.env`, bản sao `.env.prod`), khác DB devtest (`.env.devtest`, trước đây là `.env`) dùng cho snapshot `2026-11-18`. Kiểm tra read-only bằng `scripts/check_db.py --env .env.prod` (so theo tháng với snapshot).

- **DB production đang cập nhật:** ngày thường khoảng 2.700 CSYT có số, cuối tuần khoảng 350–400. Ngày đủ cuối cùng là 23/11/2026; 24/11 mới có 185 CSYT (trễ khoảng 2 ngày). DB cũ dừng ở 18/06/2026.
- **Trùng với snapshot tới 06/2024:** `tong` theo tháng lệch dưới 0,03%; DB production có thêm 5–80 CSYT, toàn CSYT số liệu rất nhỏ.
- **Từ 07/2024 DB production đầy đủ hơn nhiều:** `tong` cao hơn 13–31% (10/2024 → 12/2025) và 40–67% (01 → 05/2026), thêm 470–820 CSYT mỗi tháng. Nghĩa là snapshot dùng để train thiếu một phần đáng kể dữ liệu.
- **Danh mục:** `dm_coso_kcb` 15.306 CSYT (snapshot 13.923); `config_tinh_dtm_kcb` 24 tỉnh (snapshot 1).
- **Lỗi nhỏ:** 25 dòng `tong` > 0 ở ngày tương lai (26–30/11/2026), có thể là hẹn khám hoặc nhập sai ngày; không có `csyt_id` NULL, không có giá trị âm. Có vài dòng lẻ từ 2018–2020.
- **Riêng 131 CSYT tập chặt** (so theo CSYT, 06/2025 → 07/06/2026): 120 CSYT khớp snapshot (tỉ lệ `tong` prod/snapshot trung vị 1,000, 116/120 lệch ≤ 5%). 11 CSYT Cà Mau (96004, 96011, 96014, 96019, 96025, 96034, 96066, 96067, 96145, 96146, 96155) không có số trong prod từ 06/2025 tới 09/2026, có lại từ 10/2026; chiếm khoảng 30% `tong` của tập chặt.
- **Độ đầy đủ tới nay:** 329 CSYT có số ≥ 95% ngày T2–T6 trong 24/09 → 23/11/2026; trong tập chặt chỉ 11. Từ 01/07/2026 phần lớn CSYT tập chặt của 66, 79, 80, 96 không còn số trong prod; 01, 12, 22, 24, 40, 42, 46 vẫn đều.

## 9. Đánh giá ngoài mẫu train-final bằng số thật từ DB production (25/11/2026)

Chỉ đánh giá, không train (quy ước mục 1). Dự báo `artifacts/lgbm/20261123-165545/forecast.csv` (dữ liệu tới 07/06/2026, dự báo 08/06 → 07/07/2026) so với số thật đọc từ `.env.prod`. Script tạm, chưa đưa vào `src/`.

- **Tập chấm:** 60/131 CSYT có số ≥ 95% ngày T2–T6 trong 08/06 → 07/07 (01: 18, 40: 15, 46: 10, 42: 6, 12: 6, 24: 3, 22: 1, 79: 1), 148 chuỗi, 4.440 điểm. Ngày không có dòng tính là 0 (cuối tuần/đóng cửa). Không có ngày lễ trong khoảng này.
- **Ngày nạp dở 30/06/2026:** chỉ 27/120 CSYT có số; tổng `tong` của 60 CSYT là 730 so với q50 3.479. Bỏ ngày này khi chấm.

| (bỏ 30/06) | lgbm | snaive7 |
|---|---|---|
| **WAPE** | **0,259** | 0,287 |
| WAPE h 15–30 | 0,325 | 0,343 |
| bias | −0,042 | |
| cover80 (đã hiệu chỉnh A5) | 0,825 | |

- Giữ 30/06: lgbm 0,290, snaive7 0,325, cover80 0,805. Theo chỉ tiêu (giữ 30/06): `tong` 0,272, `luot_kham` 0,295, `noi_tru` 0,293, `cap_cuu` 0,354, `ngoai_tru` 0,692.
- Khớp với test (mục 6: WAPE 0,256, cover80 0,82). Hạn chế: một gốc, chỉ 60 CSYT ở 7 tỉnh phía Bắc/Trung, không có ngày lễ.

## 10. Snapshot DB production 2026-11-25

- `extract` từ `.env` (DB prod) ngày 25/11/2026, ~20 phút. Dữ liệu ngày 2018 → 2027 (dòng 0 đệm tới hết 2027); CSYT có số: 2018–2020 chỉ vài CSYT, 2021 ~180, 2023 ~680, 2025 ~4.000, 2026 ~4.200. 15.306 CSYT trong `dm_coso_kcb`.
- Ngày cuối đủ số: **23/11/2026**. 24/11 chỉ 202 CSYT báo cáo (7% mức thường, trễ ETL ~2 ngày); 25/11 (ngày extract) 1.840 CSYT nhưng tổng `tong` 14 nghìn (~18% một ngày thường); 26–30/11 vài chục dòng ngày ra sai.
- `clean.load_raw` bỏ mọi ngày từ ngày extract (tên snapshot) trở đi, nên `last_data_day` = 23/11. Snapshot cũ 2026-11-18 không đổi (07/06/2026).
- Dữ liệu trước 01/01/2021 bị bỏ trong `load_raw`: chỉ 10 CSYT tỉnh 40 với 285 lượt rải rác 2018–2020; lịch nghỉ `calendar/` bắt đầu từ 2021 (snapshot cũ cũng bắt đầu 01/01/2021).
- `quality` với ngưỡng gốc (cửa sổ 365 ngày tới 23/11/2026): **tập chặt 26 CSYT, tập nới 38** (snapshot cũ: 131/204). Chỉ 4/131 CSYT của tập chặt cũ còn đạt; phần lớn rơi ở coverage (median 0,85) và gap (median 20 ngày làm việc), kèm collapse.
- Nguyên nhân là các lỗ nạp dữ liệu theo tỉnh trong DB prod, không phải CSYT ngừng hoạt động (số CSYT báo cáo trung bình theo ngày làm việc):
  - 01–29/07/2026: tỉnh 66, 79, 86, 96, 75, 80 gần như im lặng (66: 287 → 25, 79: 221 → 11), 44 còn ~60%; các tỉnh khác bình thường.
  - 28/09–19/10/2026: gần như toàn hệ thống (2.898 → 684 CSYT/ngày), chỉ 24, 33 bình thường.
  - Ngày cuối/đầu tháng nạp dở: 31/12/2025–02/01/2026, 27–30/01/2026, 31/03/2026.
  - Tỉnh 51 dừng hẳn từ 24/09/2026; 11 CSYT Cà Mau không có số 06/2025–09/2026 (đã biết từ mục 9).
- **Xử lý (user chọn 25/11/2026): đánh dấu lỗ nạp theo tỉnh** (`clean.etl_holes`). Ngày làm việc mà số CSYT báo cáo của tỉnh < 50% max cùng thứ của 8 tuần trước (max thay median để lỗ ~4 tuần không tự kéo tham chiếu xuống) là lỗ; T7/CN là lỗ khi T6 trước và T2 sau đều lỗ (số đếm cuối tuần nhiễu); ngày nghỉ và cửa sổ Tết không tính. Ngày lỗ: mask khi train (mọi CSYT trong tỉnh, kể cả CSYT có số), bỏ khỏi coverage/gap; tổng tháng/năm = trung bình ngày ngoài lỗ × số ngày, bỏ tháng lỗ quá nửa. Ngưỡng `quality` giữ nguyên.
  - Số ngày làm việc bị lỗ trong cửa sổ: 66/75/79/86: 43–44, 80: 71, 19: 81, 04: 53, 12: 44, 44: 43, 25: 43, 01/40/46: 24, 33: 4, 24: 7.
  - Trong lỗ dài (07/2026, 10/2026) số CSYT giảm 70–90% nhưng tổng lượt gần như không giảm (tỉnh 25: 12 nghìn so với 15 nghìn/ngày): bệnh viện vẫn nạp, thiếu chủ yếu là CSYT nhỏ. Ngày nạp dở cuối tháng (vd. 30/06) thì cả CSYT có số cũng thiếu (mục 9), nên mask cả tỉnh cho chắc.
  - Lỗ dài hơn 8 tuần thành "mức mới" và không còn bị đánh dấu (tỉnh 25 01–04/2026, 80 01–07/2026: số CSYT còn ~1/4 và ~1/2).
- **Kết quả: tập chặt 336 CSYT, tập nới 345** (tuyến 2: 20, 3: 78, 4: 236). So với tập chặt cũ 131: giữ 79, bỏ 52 (23 CSYT Cà Mau 96 không có số, 10 dừng hoạt động, còn lại level shift/ngoại lai/gap), thêm 257 (177 tuyến 4, median `tong` ~440 lượt/tháng; 200 CSYT đã có trong report cũ nhưng trượt coverage ở cửa sổ 06/2025–06/2026).
- `facilities`: 345 CSYT (336 tập chặt); bán kính khí tượng 6 km: 325, 12 km: 20; hạng 4: 230, 3: 68, CH: 22, 2: 12, 1: 3, unknown: 10.
- `weather` (chuẩn bị): mọi ô lưới ERA5-Land của 345 CSYT (bán kính 6 và 12 km) nằm trong khung đã tải `data/weather/22.8_103_8.3_109.7`, nên `weather` dùng lại khung này (`existing_area`) và chỉ tải thêm 08–11/2026 (`weather.end` = 2026-11; tháng hiện tại xin tới hôm nay − 6 ngày, file tải trước cuối tháng + 10 ngày sẽ được tải lại).
- `weather` (25/11/2026): dùng lại khung [22,8; 103,0; 8,3; 109,7], chỉ tải 4 file 08–11/2026 (~20 phút, CDS có lỗi mạng nhưng tự thử lại). `weather_daily`: 345 CSYT × 2 bán kính, 02/12/2020 → **19/11/2026** (ngày 20–23/11 chưa có quan trắc), thiếu `t_mean` 0,29% = chỉ CSYT 45105 (BV Mắt Quảng Trị, thuộc tập chặt; tọa độ trong DB rơi vào ô biển gần đảo Cồn Cỏ, như snapshot cũ). `features` sẽ bỏ CSYT này nên tập train còn 335 CSYT chặt. Mưa khí hậu/quan trắc 0,93. Trùng khớp hoàn toàn với `weather_daily` cũ trên phần chung (488.048 dòng, lệch < 1e-13).

## 11. Quyết định 26/11/2026 và Optuna trên snapshot prod 2026-11-25

- **Người dùng chốt:** tạm giữ ERA5 (bộ feature A3 đầy đủ); bỏ hướng TiDE khỏi production (theo cổng A4, mục 4–5; code và kết quả giữ để tái lập báo cáo), theo đó bỏ luôn TFT. Chỉ còn LightGBM.
- `features` dựng trên snapshot 2026-11-25 (tập chặt 336, bỏ CSYT 45105 thiếu khí tượng → 335).
- Optuna chạy lại từ đầu, study riêng `lgbm_2026-11-25` trong `runs/optuna.db` (study `lgbm` cũ = snapshot 2026-11-18, mục 3; WAPE trên dữ liệu khác nhau không so được). Không gian tham số như A1 (chỉ tweedie), trial 0 = config hiện tại, 20 trial, 3 gốc validation giữ nguyên (15/01/2025, 30/09/2025, 30/11/2025) để so được với phase 1. Log `runs/tune_2026-11-25.log`, bảng `runs/optuna_lgbm_2026-11-25.csv`.
- **Kết quả Optuna (26/11/2026)**: 20 trial, ~1 h 50 phút (2–16 phút/trial). Mẫu train theo gốc: 2,08 / 4,56 / 5,26 triệu; điểm đánh giá 9.390 / 23.620 / 23.929. WAPE validation gộp 3 gốc: **trial 0 (config hiện tại) 0,2882**, **tốt nhất trial 18: 0,2803** (−0,79 điểm, vượt ngưỡng 0,5 điểm của A1). Tham số trial 18: rounds 200, learning_rate 0,0211, num_leaves 163, min_data_in_leaf 75, feature_fraction 0,840, bagging_fraction 0,967, lambda_l2 0,003, tweedie_variance_power 1,613.
  - Vùng tốt nhất ổn định chứ không phải một trial may mắn: 7 trial đứng đầu (0,280–0,284) đều có ít vòng (200–300), learning_rate × rounds 4–12 (config hiện tại 30), num_leaves 94–222, lambda_l2 ≈ 0. Các trial nhiều vòng (≥ 700) đều kém hơn config hoặc ngang config.
  - WAPE 0,288 không so được với 0,258 của snapshot cũ (mục 3): tập CSYT khác (335 so với 131, thêm 177 CSYT tuyến 4 nhỏ).
- **Config mới (26/11/2026):** người dùng duyệt đưa trial 18 vào `configs/base.yaml` (rounds 200, các tham số trên; interval_rounds 200 giữ nguyên) và chạy tiếp backtest → calibrate → train-final trên snapshot 2026-11-25.
- **Sửa RAM lgbm (26/11/2026):** backtest đầy đủ trên snapshot prod hai lần bị dừng vì máy hết RAM (15 GB) ở gốc 31/01 và 02/03/2026. Nguyên nhân: ma trận mẫu 7 triệu dòng × 86 cột bị chép nhiều lần (`pd.concat`, `X[train]`, mỗi model q10/q50/q90 chuyển lại sang float64 vì lat/lon float64). Sửa: lat/lon float32, dựng `X` từng cột, một `lgb.Dataset` cho cả 3 model. Đỉnh RAM gốc 01/05/2026 ~11 GB → 6,9 GB, train-final (8,9 triệu mẫu) 8,35 GB. Dự báo giống hệt bản cũ (lệch 0 trên dữ liệu smoke).
- **Backtest snapshot prod** (`runs/20261126-170023_snaive7-mean4-lgbm-prophet_prod-prophet_clean`, config trial 18; 335 CSYT, 819 chuỗi tập chặt; gốc val/test như mục 2): lgbm 3–14 phút/gốc, Prophet ~1 phút/gốc. mean4 thiếu 15 điểm (chuỗi chưa đủ lịch sử), các model khác đủ.

  | WAPE (val / test) | tổng | h 1–7 | h 8–14 | h 15–30 | Tết ±21 | tổng tỉnh × chỉ tiêu |
  |---|---|---|---|---|---|---|
  | **lgbm (trial 18)** | **0,280 / 0,264** | 0,211 / 0,238 | 0,294 / 0,245 | 0,301 / 0,292 | 0,298 / 0,343 | 0,183 / 0,164 |
  | lgbm (config phase 1) | 0,287 / 0,271 | 0,212 / 0,238 | 0,294 / 0,257 | 0,313 / 0,301 | 0,312 / 0,377 | 0,189 / 0,163 |
  | mean4 | 0,324 / 0,378 | | | | 0,576 / 0,594 | 0,225 / 0,299 |
  | prophet_clean | 0,395 / 0,402 | | | | 0,634 / 0,591 | 0,299 / 0,310 |
  | prophet_prod | 0,417 / 0,425 | | | | 0,649 / 0,596 | 0,327 / 0,338 |
  | snaive7 | 0,374 / 0,562 | | | | 0,557 / 0,736 | 0,250 / 0,481 |

  - Trial 18 tốt hơn config phase 1 cả trên test (−0,7 điểm, chủ yếu h ≥ 8 và Tết), không chỉ trên validation nơi nó được chọn. Run so sánh: `runs/20261126-185937_lgbm_cfg-phase1` (cùng code, chỉ đổi tham số).
  - Khoảng tin cậy không kém đi dù q10/q90 dùng chung tham số mới: pinball 0,104 / 0,089 so với 0,106 / 0,090; cover80 trước hiệu chỉnh 0,757 / 0,765 so với 0,734 / 0,733.
  - Wilcoxon so với ProphetProd: lgbm tốt hơn ở 97% chuỗi test, 87% chuỗi val (p < 1e-4).
  - Theo chỉ tiêu (test): `tong` 0,245, `luot_kham` 0,249, `ngoai_tru` 0,394, `cap_cuu` 0,601, `noi_tru` 0,885 (snaive7 0,768: nội trú test vẫn yếu). Ngày lễ: lgbm 0,664 / 0,951, các baseline 7–17.
  - Không so được với số phase 1 (mục 3, test 0,256): tập CSYT và dữ liệu khác.
- **calibrate (A5)**: hệ số lo/hi theo h 1–7: 1,003 / 0,901; 8–14: 1,048 / 1,468; 15–30: 1,060 / 1,342. cover80 test 0,789 / 0,751 / 0,761 → 0,778 / 0,845 / 0,836 (val sau hiệu chỉnh 0,80).
- **train-final**: `artifacts/lgbm/20261126-184100`, train tới 23/11/2026, dự báo 24/11 → 23/12/2026; 335 CSYT, 819 chuỗi, 8,89 triệu mẫu, 1.116 giây; git 0fd8988, `git_dirty` false; hệ số khoảng tin cậy từ backtest trên. Tổng q50 `tong` ngày thường 23–32 nghìn, cuối tuần ~3,2 nghìn.
  - 24/11/2026 dự báo `tong` 1.355 vì lịch nghỉ (`holidays` 0.108) ghi "Ngày Văn hóa Việt Nam" 24/11 là ngày nghỉ quốc gia từ 2026. Dữ liệu khớp: 24/11 (đã qua trọn ngày) chỉ ~200 CSYT báo cáo, ít hơn cả ngày extract 25/11 (1.840, nạp dở), nên ở mục 8 và 10, 24/11 nhiều khả năng là ngày nghỉ chứ không phải trễ ETL. Đây là lần đầu có ngày nghỉ này: model dự báo như một ngày lễ quốc gia một ngày.
- **Dashboard** (26/11/2026): `dashboard/build_data.py` mặc định đọc artifact `train-final` mới nhất (`artifacts/lgbm/`) thay vì `lgbm_far`: dữ liệu prod đã tới 23/11/2026 nên dự báo nối liền số thật, không còn khoảng trống, không cần train-far (vẫn xem được bằng `build_data.py artifacts/lgbm_far/<run>`). `data.js` từ `20261126-184100`: 335 CSYT, thực tế 01/01/2025 → 23/11/2026, dự báo 24/11 → 23/12/2026; tổng q50 `tong` 7 ngày đầu 1.352 (lễ 24/11) · 26.835 · 25.759 · 26.311 · 3.205 · 3.004 · 29.462, tuần thật cuối 25.261 · 21.680 · 22.278 · 23.953 · 3.312 · 3.091 · 27.266.
- **Báo cáo tổng A7 trên snapshot prod** (27/11/2026): `runs/20261127-084610_report` (`report.md`, `tet_tuyen.png`) từ run backtest `20261126-170023_...` và artifact `20261126-184100`. `report.py` chấm trên phần giao các model (bỏ 15 điểm validation mean4 thiếu dự báo); TiDE và ablation là run của snapshot cũ nên không gộp (`report.tide: []`, `ablation: {}` trong `base.yaml`, cấu hình phase 1 ghi ở comment). `docs/report_summary.md` viết lại theo số prod. Báo cáo phase 1 giữ ở `runs/20261123-170738_report`.
