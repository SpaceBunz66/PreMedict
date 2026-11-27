# forecast_v2

Dự báo lượt khám 30 ngày cho CSYT. Đọc DB chỉ đọc, làm sạch, ghép lịch lễ, rồi backtest và train LightGBM. Khí tượng ERA5-Land, TiDE và Prophet chỉ để so sánh trong báo cáo: production không dùng ERA5 (`features.drop` trong `configs/base.yaml`).

| Thư mục | Nội dung |
|---|---|
| `configs/base.yaml` | mọi tham số: dữ liệu, feature, gốc backtest, model, run dùng cho báo cáo |
| `src/forecast_v2/` | các bước pipeline (`cli.py` là điểm vào); `models/`: lgbm, TiDE, Prophet, baseline |
| `calendar/vn_holidays.csv` | lịch nghỉ lễ Việt Nam |
| `tests/` | `pytest` |
| `docs/` | `report_config.md` (cấu hình, số liệu từng bước), `report_summary.md` (kết luận) |
| `dashboard/` | demo: `build_data.py` → `data.js`, mở `index.html` |
| `data/`, `runs/`, `artifacts/` | snapshot dữ liệu, kết quả backtest, model đã train. Chỉ ở local, không commit |

Cài đặt: `.venv-ml` theo `requirements-ml.txt`, rồi `pip install -e .`. Kết nối DB lấy từ `.env` (DB production; `extract --env <file>` để đọc DB khác, vd. `.env.devtest`), tải ERA5 dùng `.cdsapirc`.

```
python -m forecast_v2.cli extract | quality | facilities | weather | features   # dựng snapshot trong data/; weather chỉ cần cho báo cáo
python -m forecast_v2.cli backtest --models snaive7,mean4,lgbm                  # -> runs/<giờ>_<model>/metrics.md
python -m forecast_v2.cli calibrate --models lgbm                               # A5: hệ số khoảng tin cậy
python -m forecast_v2.cli train-final --models lgbm                             # -> artifacts/lgbm/<giờ>/
python -m forecast_v2.cli train-far                                             # dự báo từ hôm nay -> artifacts/lgbm_far/<giờ>/
python -m forecast_v2.cli report                                                # -> runs/<giờ>_report/report.md
python -m forecast_v2.cli evaluate --artifact artifacts/lgbm/<giờ>              # chấm bằng số thật snapshot mới -> eval_<snapshot>.md
python dashboard/build_data.py                                                  # demo, đọc artifacts/lgbm mới nhất (train-final)
```

Kết quả mới nhất: báo cáo `runs/*_report/report.md`, model `artifacts/lgbm/` (30 ngày sau ngày cuối có dữ liệu) và `artifacts/lgbm_far/` (từ hôm nay), dashboard `dashboard/index.html`. Lệnh đầy đủ và cờ khác: docstring của `src/forecast_v2/cli.py`.
