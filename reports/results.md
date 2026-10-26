# Kết quả thực nghiệm

Mô hình được chọn theo validation macro-F1; test chỉ đánh giá mô hình được chọn.

| model             |   train_f1_macro |   validation_f1_macro |   f1_macro_gap |   test_accuracy |   test_balanced_accuracy |   test_f1_macro |   test_f1_weighted |
|:------------------|-----------------:|----------------------:|---------------:|----------------:|-------------------------:|----------------:|-------------------:|
| Extra Trees       |           0.9773 |                0.8018 |         0.1755 |          0.7635 |                   0.7984 |          0.8021 |             0.7662 |
| SVM               |           0.8583 |                0.7826 |         0.0756 |                 |                          |                 |                    |
| Random Forest     |           0.9871 |                0.7787 |         0.2085 |                 |                          |                 |                    |
| Gradient Boosting |           0.8703 |                0.7437 |         0.1266 |                 |                          |                 |                    |
| k-NN              |           0.8026 |                0.7208 |         0.0817 |                 |                          |                 |                    |
