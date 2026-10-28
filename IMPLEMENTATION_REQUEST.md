# Yêu cầu triển khai (Implementation Request)

Ghi chú: **KHÔNG commit tự động.** Tài liệu này chỉ để AI khác đọc và triển khai sau.

## Mục tiêu
Thay đổi quy trình huấn luyện để tuân thủ đúng yêu cầu đề tài:

1. Chia dữ liệu 70–15–15, có stratify=y.
2. Huấn luyện 5 mô hình với cấu hình ban đầu ở trên.
3. Dùng 5-Fold Cross-validation trên tập Training để tinh chỉnh tham số.
4. Dùng Validation để kiểm tra và lựa chọn cấu hình cuối.
5. Huấn luyện lại mô hình đã chọn trên dữ liệu Training + Validation, sau đó đánh giá một lần trên Test.
6. Báo cáo Accuracy, Macro F1, Weighted F1, Classification Report và Confusion Matrix.
7. Thực hiện thí nghiệm bổ sung cho SVM + OvR và Gradient Boosting + Early Stopping nếu thời gian cho phép.
Lưu ý: Khi chọn cấu hình bằng Cross-validation, đừng dùng tập Test để điều chỉnh tham số. Nếu dùng Early Stopping trong từng fold, việc chọn điểm dừng phải dựa trên dữ liệu huấn luyện của fold đó, không dùng Test.

## Trạng thái hiện tại
- Chia: train/val/test theo stratified split (60/20/20) với `TEST_SIZE=0.2`, `VALIDATION_SIZE=0.2` (đã có stratify). Cần đảm bảo đúng tỷ lệ 70/15/15 theo đúng nghĩa đề bài.
- Chọn mô hình theo **hold-out validation** (`validation_f1_macro`).
- **Không có 5-Fold CV để tinh chỉnh tham số**.
- SVM đã dùng OvR. Gradient Boosting đã có `n_iter_no_change`, `validation_fraction`, `tol` (early stopping nội bộ).

## Đề xuất thay đổi

### 1) Chia dữ liệu
- Thay đổi thành **70–15–15** (train/validation/test = 70/15/15) với `stratify=y`.
- Giữ logic stratified split như hiện tại (hai lần stratified train_test_split) hoặc cấu hình lại.
- Đảm bảo Test **không** được dùng trong quá trình tinh chỉnh/tìm kiếm tham số.

### 2) Tinh chỉnh tham số bằng 5-Fold CV trên tập **Training**
- Với mỗi mô hình, sử dụng `GridSearchCV` hoặc `RandomizedSearchCV` với **cv=5** trên **chỉ tập X_train**.
- Tìm cấu hình tốt nhất theo **macro-F1** (hoặc metric phù hợp).
- **Không** truy cập X_test/X_validation để tìm kiếm.
- Ghi lại kết quả CV (mean/std của từng cấu hình) nếu có thể.

### 3) Lựa chọn cấu hình cuối bằng Validation
- Sau khi có cấu hình tốt nhất từ CV trên train, huấn luyện (hoặc đánh giá) mô hình đó trên **X_train** để so số liệu với **X_validation** (hold-out) như yêu cầu “Dùng Validation để kiểm tra và lựa chọn cấu hình cuối”.
- Quy trình này nhằm đảm bảo có sự so sáy độc lập giữa các ứng viên sau tinh chỉnh.

### 4) Huấn luyện lại và đánh giá Test
- Lấy mô hình/cấu hình **đã chọn** nhờ validation, huấn luyện lại trên **X_train + X_validation**.
- Đánh báo một lần trên **X_test**.
- Báo cáo đầy đủ: Accuracy, Macro F1, Weighted F1, Classification Report, Confusion Matrix.

### 5) Bổ sung thí nghiệm
- SVM: đảm bảo sử dụng **SVC + OneVsRestClassifier (OvR)**.
- Gradient Boosting: nếu triển khai CV trong từng fold, cần lưu ý rằng early stopping nội bộ của `GradientBoostingClassifier` dùng `validation_fraction` từ dữ liệu **của fold đó** (không dùng Test). 
- Có thể áp dụng GridSearchCV/parameter grid cho SVM (C, gamma, kernel) và Gradient Boosting (n_estimators, learning_rate, max_depth, subsample, n_iter_no_change, validation_fraction).

## Các tập tin cần chỉnh sửa
- `src/config.py`: thay đổi `TEST_SIZE`, `VALIDATION_SIZE` thành 0.15/0.15 (hoặc cấu hình phù hợp 70/15/15).
- `src/data.py`: đảm bảo chia đúng tỷ lệ 70–15–15 với stratify; có thể điều chỉnh logic split nếu cần.
- `src/models.py`: có thể thêm `param_grid` hoặc function hỗ trợ CV.
- `src/train.py`: 
  - Thay đổi quy trình: CV trên train, đánh giá trên validation để chọn, fit lại trên train+validation, đánh giá test.
  - Lưu CV results, classification reports, confusion matrices.
  - Đảm bảo không lộ test trong CV.

## Lệnh test đề nghị
- `python -m src.train --push` hoặc các trường hợp khác, nhưng **KHÔNG commit** tự động khi triển khai — chỉ lưu kết quả ra file.
이드
