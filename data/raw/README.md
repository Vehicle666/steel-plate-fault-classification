# Dữ liệu gốc — Steel Plates Faults (UCI)

Nguồn: <https://archive.ics.uci.edu/dataset/198/steel+plates+faults>

| File | Mô tả |
|---|---|
| `steel+plates+faults.zip` | File gốc tải từ UCI (giữ nguyên để đối chiếu) |
| `steel_plates_faults/Faults.NNA` | Dữ liệu: 1941 dòng × 34 cột, phân tách bằng tab, **không có header** |
| `steel_plates_faults/Faults27x7_var` | Tên 34 cột: 27 đặc trưng + 7 cột nhãn one-hot |

## Cấu trúc

- **27 đặc trưng đầu**: hình học (`X_Minimum`, `X_Maximum`, `Y_Minimum`, `Y_Maximum`, `Pixels_Areas`, các chỉ số perimeter/index...) và độ sáng (`Sum_of_Luminosity`, `Minimum_of_Luminosity`, `Maximum_of_Luminosity`, `Luminosity_Index`...).
- **7 cột cuối** là one-hot của 7 lớp lỗi, đúng **một** cột bằng 1 mỗi dòng:
  `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`.

> Lưu ý: cột `TypeOfSteel_A300` và `TypeOfSteel_A400` là biến nhị phân (một trong hai bằng 1).

## Tải lại

```powershell
.\.venv\Scripts\python.exe -m src.download_data --force
```

## Trích dẫn

> Buscema, M. (2013). Steel Plates Faults [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5J590

Giấy phép: CC BY 4.0 (theo UCI ML Repository).
