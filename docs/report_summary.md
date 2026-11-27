# Tóm tắt và kết luận

Mục tiêu: dự báo 30 ngày tới số lượt khám mỗi ngày của 5 chỉ tiêu (`tong`, `luot_kham`, `ngoai_tru`, `noi_tru`, `cap_cuu`) cho 335 CSYT của tập chặt (819 chuỗi), thay lõi Prophet đang chạy production. Dữ liệu: snapshot DB production 2026-11-25, đủ số tới 23/11/2026 (phụ lục mục 10–11). Mọi lựa chọn dựa trên 3 gốc validation; 5 gốc test chỉ để báo cáo. WAPE = tổng |sai số| / tổng thực tế, càng nhỏ càng tốt.

## Kết luận

1. **Chọn LightGBM** (direct multi-horizon, một model cho mọi chuỗi và mọi h).
   - WAPE test 0,262 (validation 0,280). ProphetProd đang chạy: 0,425 (0,417). Giảm 16,3 điểm WAPE trên test.
   - Tốt hơn ProphetProd ở 97% chuỗi trên test và 86% trên validation (Wilcoxon, p < 0,0001).
   - Cộng lên tổng tỉnh × chỉ tiêu: 0,163 so với 0,338. Trong cửa sổ Tết ±21 ngày: 0,344 so với 0,596. ProphetProd dự báo như ngày thường suốt kỳ nghỉ Tết (biểu đồ Tết 2026).
   - Sai số tăng dần theo horizon: 0,235 (ngày 1–7), 0,240 (8–14), 0,293 (15–30).
   - Theo tuyến (test): tuyến 3 0,240, tuyến 2 0,291, tuyến 4 0,299 (236 CSYT nhỏ, chiếm 60% số điểm). Tuyến 4 dự báo thiếu đợt tăng trước Tết (09–13/02/2026: thực tế ~8 nghìn/ngày, q50 ~6 nghìn).
2. **Lợi ích chủ yếu đến từ model, không phải từ làm sạch dữ liệu.** Cùng Prophet nhưng chạy trên dữ liệu đã làm sạch (prophet_clean) chỉ giảm WAPE test 2,3 điểm (0,402). Đổi sang LightGBM giảm thêm 14,0 điểm.
3. **Lịch lễ Tết Việt Nam là feature quyết định.** Nhóm lễ, Tết, âm lịch chiếm 57% gain của LightGBM; riêng `is_day_off` chiếm 52%. Ngày lễ: WAPE test 0,97 so với 15–17 của Prophet. Ablation phase 1 (snapshot cũ, 131 CSYT): bỏ nhóm này làm WAPE kém 6,7 điểm trên validation, 8,2 điểm trên test.
4. **Khí tượng ERA5 không giúp, production bỏ ERA5** (người dùng chốt 27/11/2026; bộ feature A1: lịch + lễ Tết, ERA5 chỉ giữ trong báo cáo). Trên snapshot prod, cùng code và tham số: bỏ ERA5 cho WAPE validation 0,280 so với 0,280, test 0,262 so với 0,264, Tết ±21 test 0,344 so với 0,343 (bảng ablation). Khi có ERA5, khí tượng quan trắc chỉ chiếm 0,4% gain, khí hậu 1,0%. Ablation phase 1 cũng vậy: chênh dưới 0,3 điểm và đổi dấu giữa validation và test.
5. **Tham số Optuna mới (trial 18).** Tune lại trên snapshot prod (khi còn ERA5): validation 0,2882 → 0,2803. Xác nhận trên test với cùng code, chỉ đổi tham số: 0,271 → 0,264, tốt hơn chủ yếu ở h ≥ 8 và quanh Tết (0,377 → 0,343).
6. **Bỏ TiDE và TFT khỏi production** theo cổng A4 (phase 1): TiDE s2000 WAPE validation 0,299, test 0,273, kém LightGBM, hỏng ở ngày lễ và chậm gấp khoảng 11 lần; trộn với LightGBM không giúp. Người dùng chốt 26/11/2026.
7. **Khoảng tin cậy 80% đã hiệu chỉnh (A5).** Độ phủ trên test tăng từ 0,756 lên 0,821 sau khi giãn khoảng theo hệ số fit trên validation.
8. **Model production**: `artifacts/lgbm/20261127-123618` (không ERA5). Train trên toàn bộ dữ liệu tới 23/11/2026 (8,9 triệu mẫu, ~17 phút), dự báo 24/11 → 23/12/2026, đã áp hệ số A5. Kiểm tra ngoài mẫu của model phase 1 bằng số thật prod (08/06 → 07/07/2026, 60 CSYT): WAPE 0,259, khớp mức test.

## Còn chờ quyết định

- **Pha production (§8 plan):** dự báo cấp tỉnh (cộng dồn từ CSYT hay giữ Prophet cũ), chuyển đổi (tắt service cũ hay chạy song song), và fallback cho CSYT ngoài tập chặt (đang chọn tạm giữ Prophet).

## Hạn chế và việc còn lại

- Tập chặt được lọc theo chất lượng của 365 ngày cuối snapshot (24/11/2025 → 23/11/2026), trùng một phần giai đoạn test. So sánh giữa các model vẫn công bằng (cùng tập), nhưng WAPE tuyệt đối có thể hơi lạc quan.
- DB prod có lỗ nạp dữ liệu theo tỉnh (07/2026, 28/09–19/10/2026, ngày cuối tháng): đã mask khi train và chấm (`clean.etl_holes`, phụ lục mục 10). Production cần cùng cách xử lý, nếu không model sẽ học các số 0 giả.
- `noi_tru`, `cap_cuu` và `ngoai_tru` là các chuỗi nhỏ và nhiễu: WAPE test của LightGBM là 0,88, 0,60 và 0,39; với `noi_tru`, snaive7 (0,77) còn tốt hơn.
- 24/11/2026 (Ngày Văn hóa Việt Nam) lần đầu là ngày nghỉ theo lịch `holidays` 0.108; model dự báo như một ngày lễ quốc gia, dữ liệu prod khớp. Cần kiểm tra lại khi có số thật.
- Chưa làm:
  - A6: kiểm tra trên tầng nới và quy tắc fallback cho CSYT thiếu dữ liệu;
  - ablation bỏ nhóm lễ Tết (A2) trên snapshot prod (mới chạy lại phần ERA5);
  - service production (§8 plan), `WRITE_DB` tắt;
  - TFT (bỏ theo A4).
