# Phân loại lỗi bề mặt tấm thép

Phân loại **7 loại lỗi bề mặt tấm thép** từ **đặc trưng hình học và độ sáng**, sử dụng các mô hình học máy cổ điển với chiến lược **One-vs-Rest (OvR)**.

## Thông tin đề tài

| Mục | Nội dung |
|---|---|
| Tên đề tài | Phân loại loại lỗi bề mặt tấm thép từ đặc trưng hình học và độ sáng |
| Mô hình | SVM, k-NN, Random Forest, Extra Trees, Gradient Boosting |
| Chiến lược đa lớp | One-vs-Rest (OvR) |
| Đánh giá | Accuracy, Macro/Weighted F1, Ma trận nhầm lẫn (confusion matrix) |
| Dataset | [Steel Plates Faults – UCI ML Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults) |
| Quy mô | 1.941 mẫu, 27 đặc trưng, 7 lớp lỗi |
| Mức độ | Trung bình |

### Lưu ý về dữ liệu

- Nhãn gốc ở dạng **one-hot 7 cột** → cần chuyển về **một nhãn duy nhất** (`argmax`).
- 7 lớp: `Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`.
- Lớp **`Other_Faults`** gom nhiều dạng lỗi khác nhau nên rất khó tách -> cần phân tích riêng trong confusion matrix.

## Cấu trúc project

```
.
├── data/
│   ├── raw/              # dữ liệu gốc tải từ UCI (không commit)
│   └── processed/        # dữ liệu sau tiền xử lý (không commit)
├── models/               # mô hình đã huấn luyện (không commit)
├── reports/
│   └── figures/          # biểu đồ, confusion matrix (không commit)
├── src/                  # mã nguồn
├── requirements.txt
└── README.md
```

## Cài đặt môi trường

```powershell
# 1. Tạo virtual environment
python -m venv .venv

# 2. Kích hoạt
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt thư viện
pip install -r requirements.txt
```

## Trạng thái

- [x] Khởi tạo môi trường & cấu trúc project
- [ ] Tải và khảo sát dataset
- [ ] Tiền xử lý (one-hot -> nhãn, chuẩn hoá, chia train/test)
- [ ] Huấn luyện 5 mô hình với OvR
- [ ] Đánh giá & confusion matrix
- [ ] Báo cáo kết quả
