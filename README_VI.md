# QTRC-Stop cho tập Polliradar

Repo triển khai bài toán phân loại taxonomic sớm từ tín hiệu mmWave `amplitude/phase`. Mỗi quyết định sử dụng prefix nhân quả 0,1--2,0 s; train, calibration và test tách theo `subject_id`. Policy QTRC-Stop kết hợp posterior đã hiệu chỉnh, độ ổn định giữa các prefix, harmonic ratio, harmonic-energy ratio và độ ổn định của `F0`.

## 1. Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e . --no-build-isolation
PYTHONPATH=src pytest -q
# PowerShell: $env:PYTHONPATH="src"; pytest -q
```

Kết quả test phải là `4 passed`.

## 2. Chạy

Linux, macOS, WSL hoặc Git Bash:

```bash
bash scripts/run_smoke.sh "/path/Polliradar dataset.zip" outputs/smoke
bash scripts/run_pilot.sh "/path/Polliradar dataset.zip" outputs/pilot
bash scripts/run_full.sh  "/path/Polliradar dataset.zip" outputs/full
```

Lệnh Python dùng được trên mọi hệ điều hành:

```bash
python -m qtrc_stop.cli run \
  --config configs/full.yaml \
  --dataset "/path/Polliradar dataset.zip" \
  --run-dir outputs/full
```

PowerShell:

```powershell
.\scripts\run_smoke.ps1 -Dataset "D:\data\Polliradar dataset.zip" -RunDir "outputs\smoke"
.\scripts\run_full.ps1  -Dataset "D:\data\Polliradar dataset.zip" -RunDir "outputs\full"
```

## 3. Checkpoints

- **CP0**: giải nén và kiểm kê 167 CSV; chuẩn hóa `bombus_lapidaries` thành `bombus_lapidarius`.
- **CP1**: phát hiện event micro-Doppler theo recording.
- **CP2**: trích xuất 86 đặc trưng tại bảy prefix.
- **CP3**: gate số subject, năm lớp, NaN, completeness và fallback.
- **CP4**: inner OOF và outer test prediction, tách theo subject.
- **CP5**: temperature scaling, taxonomic loss và policy grid.
- **CP6**: fixed-duration, raw/calibrated confidence, two-stage reliability, AAGC-style và QTRC-Stop.
- **CP7**: AWGN, phase noise và amplitude sensitivity trên raw signal.
- **CP8**: bảng tổng hợp và hình.

Kiểm tra:

```bash
bash scripts/status.sh outputs/full
bash scripts/verify_run.sh outputs/full
```

## 4. Khôi phục sau mất điện

Chạy lại đúng lệnh cũ:

```bash
bash scripts/resume_full.sh "/path/Polliradar dataset.zip" outputs/full
```

Tác vụ hoàn thành được nhận diện theo recording, fold và duration. Tệp được ghi qua tệp tạm rồi đổi tên nguyên tử. Lock chứa PID; lock của tiến trình đã chết được tự loại bỏ. Không xóa `state.json`, `parts/`, `models/`, `policy_grids/` hoặc `policies/`.

Cấu hình khoa học được băm. Khi thay đổi detector, feature, fold, model hoặc policy grid, dùng `run_dir` mới. Phần `robustness` có thể bật sau CP6 vì không làm thay đổi artifact CP0--CP6.

## 5. Synthetic robustness

```bash
bash scripts/run_robustness.sh "/path/Polliradar dataset.zip" outputs/full configs/full.yaml
```

Kết quả CP7 chỉ là sensitivity dưới nhiễu tổng hợp, không chứng minh field-domain robustness.

## 6. Phạm vi khoa học

- Python tạo 86 đặc trưng theo các nhóm harmonic, spectral, temporal, band power và MFCC. Đây không phải bản sao từng hàm của bộ 70 đặc trưng MATLAB trong bài báo nguồn.
- `two_stage_reliability` là baseline theo cấu trúc hai tầng, không phải tái tạo nguyên bản TEASER.
- `aagc_style` là confidence-only risk wrapper nội bộ, không phải triển khai chính thức thuật toán ICML.
- `bootstrap_subject` tạo cận trên thực nghiệm. Chuyển `policy.bound_method` sang `hoeffding_subject` khi cần cận phân phối-tự-do; cận có thể buộc policy quay về 2 s.
- Chỉ số theo loài là đánh giá thực nghiệm, không phải bảo đảm class-conditional hữu hạn mẫu.

Tài liệu chạy chi tiết: `guide/QTRC_Stop_Huong_dan_IEEE1cot.pdf`.
