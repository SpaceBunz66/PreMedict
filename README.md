# forecast_v2

Dự báo lượt khám 30 ngày cho CSYT. Đọc DB chỉ đọc, làm sạch, ghép khí tượng ERA5-Land và lịch lễ, rồi backtest và train LightGBM. TiDE và Prophet chỉ để so sánh.

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
python -m forecast_v2.cli extract | quality | facilities | weather | features   # dựng snapshot trong data/
python -m forecast_v2.cli backtest --models snaive7,mean4,lgbm                  # -> runs/<giờ>_<model>/metrics.md
python -m forecast_v2.cli calibrate --models lgbm                               # A5: hệ số khoảng tin cậy
python -m forecast_v2.cli train-final --models lgbm                             # -> artifacts/lgbm/<giờ>/
python -m forecast_v2.cli train-far                                             # dự báo từ hôm nay -> artifacts/lgbm_far/<giờ>/
python -m forecast_v2.cli report                                                # -> runs/<giờ>_report/report.md
python -m forecast_v2.cli evaluate --artifact artifacts/lgbm/<giờ>              # chấm bằng số thật snapshot mới -> eval_<snapshot>.md
python dashboard/build_data.py                                                  # demo, đọc artifacts/lgbm mới nhất (train-final)
```

**Job tải ERA5-Land hằng ngày** (`weather-crawl`): tải file thô vào `data/weather/<khung>/` tới hôm nay − 6 ngày (ERA5-Land trễ ~5 ngày), tải lại tháng chưa trọn, không ghi vào snapshot (`weather_daily` dựng bằng lệnh `weather` khi có snapshot mới, file đã tải thì bỏ qua). Trên laptop chạy bằng Task Scheduler, task `forecast_v2 weather-crawl` → `scripts/weather_crawl.cmd`, 09:00 hằng ngày (máy tắt lúc đó thì chạy khi bật lại), log nối vào `runs/weather_crawl.log`. Lỗi mạng/CDS: lần chạy sau tự tải tiếp.

```
Get-ScheduledTaskInfo 'forecast_v2 weather-crawl'                          # LastTaskResult 0 = thành công
Start-ScheduledTask 'forecast_v2 weather-crawl'                            # chạy ngay
Set-ScheduledTask 'forecast_v2 weather-crawl' -Trigger (New-ScheduledTaskTrigger -Daily -At '07:00')   # đổi giờ
Unregister-ScheduledTask 'forecast_v2 weather-crawl' -Confirm:$false       # gỡ job
```

Kết quả mới nhất: báo cáo `runs/*_report/report.md`, model `artifacts/lgbm/` (30 ngày sau ngày cuối có dữ liệu) và `artifacts/lgbm_far/` (từ hôm nay), dashboard `dashboard/index.html`. Lệnh đầy đủ và cờ khác: docstring của `src/forecast_v2/cli.py`.
