# Tóm tắt và kết luận

Mục tiêu: dự báo 30 ngày tới số lượt khám mỗi ngày của 5 chỉ tiêu (`tong`, `luot_kham`, `ngoai_tru`, `noi_tru`, `cap_cuu`) cho 131 CSYT của tập chặt (318 chuỗi), thay lõi Prophet đang chạy production. Mọi lựa chọn dựa trên 3 gốc validation; 5 gốc test chỉ để báo cáo. WAPE = tổng |sai số| / tổng thực tế, càng nhỏ càng tốt.

## Kết luận

1. **Chọn LightGBM** (direct multi-horizon, một model cho mọi chuỗi và mọi h).
   - WAPE test 0,256 (validation 0,2582). ProphetProd đang chạy: 0,399 (0,373). Giảm 14,3 điểm WAPE trên test.
   - Tốt hơn ProphetProd ở 95% chuỗi trên test và 87% trên validation (Wilcoxon, p < 0,0001).
   - Cộng lên tổng tỉnh × chỉ tiêu: 0,164 so với 0,323. Trong cửa sổ Tết ±21 ngày: 0,322 so với 0,569. ProphetProd dự báo như ngày thường suốt kỳ nghỉ Tết (biểu đồ Tết 2026).
   - Sai số tăng dần theo horizon: 0,234 (ngày 1–7), 0,245 (8–14), 0,277 (15–30).
2. **Lợi ích chủ yếu đến từ model, không phải từ làm sạch dữ liệu.** Cùng Prophet nhưng chạy trên dữ liệu đã làm sạch (ProphetProd-clean) chỉ giảm WAPE test 2,6 điểm (0,373). Đổi sang LightGBM giảm thêm 11,7 điểm.
3. **Lịch lễ Tết Việt Nam là feature quyết định.** Bỏ nhóm lễ, Tết, âm lịch làm WAPE kém 6,7 điểm trên validation và 8,2 điểm trên test (A0 so với A1). Sai số quanh Tết gần gấp đôi: 0,602 so với 0,321. Nhóm này chiếm 53% gain của LightGBM; riêng `is_day_off` chiếm 49%.
4. **Khí tượng ERA5 không đem lại giá trị ổn định.** Đủ bộ feature (A3) so với bỏ cả `w_*` lẫn `clim_*` (A1): validation 0,2582 so với 0,2607, test 0,2562 so với 0,2555. Chênh dưới 0,3 điểm và đổi dấu giữa hai tập. (Các run ablation chạy trước bản sửa q50 ở kết luận 7; bản sửa đổi WAPE tổng của A3 chỉ 0,02 điểm.) Khí tượng quan trắc chiếm 0,8% gain, khí hậu 2,0%.
5. **Bỏ TiDE khỏi production** theo cổng A4 (tiêu chí đã duyệt ngày 20/11/2026). Cấu hình tốt nhất (s2000) có WAPE validation 0,299, test 0,273, đều kém LightGBM. TiDE hỏng ở ngày lễ (test 2,21 so với 0,83) và chậm gấp khoảng 11 lần (2.978 so với 260 giây mỗi gốc). Trộn với LightGBM cũng không giúp. TFT không làm, cùng lý do: phần lớn chuỗi bắt đầu 2024–2025 nên mỗi chuỗi gần như chỉ có một Tết để học.
6. **Khoảng tin cậy 80% đã hiệu chỉnh (A5).** Độ phủ trên test tăng từ 0,76 lên 0,82 sau khi giãn khoảng theo hệ số fit trên validation.
7. **Model production**: `artifacts/lgbm/20261123-165545`. Train trên toàn bộ dữ liệu tới 07/06/2026, dự báo 08/06 → 07/07/2026, đã áp hệ số A5. q50 là dự báo của model tweedie: bản sửa 23/11/2026 bỏ bước sắp xếp 3 phân vị từng đẩy q50 ngày lễ lên mức ngày thường (WAPE ngày lễ validation 0,893 → 0,763).

## Còn chờ quyết định

- **Bỏ ERA5 hay không.** Đề xuất bỏ (dùng bộ feature A1) để production không cần job tải khí tượng hằng ngày, tài khoản CDS và độ trễ 5 ngày. Bản train-final hiện tại vẫn dùng đủ feature (A3). Nếu chốt bỏ thì train-final lại.
- **Tham số Optuna (A1).** Trial 16 hơn config 0,4975 điểm, sát ngưỡng 0,5 nhưng chưa đạt, nên đang giữ config.

## Hạn chế và việc còn lại

- Tập chặt được lọc theo chất lượng của 365 ngày cuối snapshot, trùng giai đoạn test. So sánh giữa các model vẫn công bằng (cùng tập), nhưng WAPE tuyệt đối có thể hơi lạc quan.
- `cap_cuu`, `noi_tru` và `ngoai_tru` là các chuỗi nhỏ và nhiễu: WAPE test của LightGBM là 0,96, 0,52 và 0,40.
- Dữ liệu ETL dừng ở 07/06/2026. Phải có dữ liệu mới trước khi lên production (bước B).
- Chưa làm:
  - A6: kiểm tra trên tầng nới khoảng 430 CSYT và quy tắc fallback cho CSYT thiếu dữ liệu;
  - lát cắt ngày thời tiết cực đoan và các chỉ tiêu suy ra;
  - đo RAM;
  - TFT.
- 4 run ablation (A2) chạy trước bản sửa q50 nên chưa khớp tuyệt đối với run chính; chạy lại mất khoảng 2 giờ 15 phút nếu cần số chính xác.
