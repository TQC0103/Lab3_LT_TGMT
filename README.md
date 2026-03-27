# Lab3-LT

## Cấu trúc thư mục

```text
.
|-- data/
|-- report/
|   |-- assets/
|   |-- appendix/
|   |-- build/
|   |-- figures/
|   |-- sections/
|   |-- tables/
|   `-- main.tex
|-- scripts/
|-- .vscode/
`-- README.md
```

## Mục đích

Project này tiếp nối `Lab2-LT`. Nếu Lab 2 tập trung vào EDA cho MNIST Handwritten, thì Lab 3 tập trung vào xây dựng lời giải phân lớp chữ số bằng ANN, không dùng CNN.

Hướng triển khai hiện tại là:

- tiền xử lý hình học bằng `deslant + recenter`
- trích đặc trưng `HOG`
- ghép `raw pixel đã chuẩn hóa hình học` và `HOG features`
- huấn luyện `ANN hai nhánh` với `BatchNorm + Dropout`

## Cách dùng nhanh

1. Cập nhật metadata trong `report/main.tex`.
2. Chỉnh lời nói đầu trong `report/preface.tex` nếu cần.
3. Viết nội dung chính trong `report/sections/content_main.tex` và các file chương liên quan.
4. Bổ sung tài liệu tham khảo vào `report/references.bib`.

## Huấn luyện mô hình ANN

Chạy baseline đầy đủ:

```powershell
python scripts\train_mnist_ann.py --epochs 20 --batch-size 256
```

Chạy baseline naive chỉ dùng raw pixel và ANN:

```powershell
python scripts\train_mnist_ann_naive.py --epochs 20 --batch-size 256
```

Chạy ANN một nhánh với `deslant + recenter + HOG + BatchNorm + Dropout`:

```powershell
python scripts\train_mnist_ann_single_branch.py --epochs 20 --batch-size 256
```

Chạy cùng pipeline trên `FashionMNIST`:

```powershell
python scripts\train_mnist_ann_single_branch.py --dataset fashionmnist --epochs 50 --batch-size 256
```

Chạy `two-branch ANN` trên `FashionMNIST`:

```powershell
python scripts\train_mnist_ann.py --dataset fashionmnist --epochs 50 --batch-size 256
```

Chạy biến thể `single-branch` với affine nhẹ hơn và augmentation cục bộ (`elastic distortion + stroke jitter`):

```powershell
python scripts\train_mnist_ann_single_branch.py --epochs 20 --batch-size 256 --cache-dir data\mnist_ann_single_branch_aug_cache --output-dir artifacts\mnist_ann_single_branch_augmented --max-rotate-deg 4 --max-shift-ratio 0.03 --max-shear-deg 3 --elastic-prob 0.45 --elastic-alpha 1.4 --elastic-sigma 4.5 --stroke-jitter-prob 0.2 --stroke-kernel-size 2
```

Chạy biến thể augmentation cục bộ trên `FashionMNIST`:

```powershell
python scripts\train_mnist_ann_single_branch.py --dataset fashionmnist --epochs 50 --batch-size 256 --cache-dir data\fashion_mnist_ann_single_branch_aug_cache --output-dir artifacts\fashion_mnist_ann_single_branch_augmented --max-rotate-deg 4 --max-shift-ratio 0.03 --max-shear-deg 3 --elastic-prob 0.45 --elastic-alpha 1.4 --elastic-sigma 4.5 --stroke-jitter-prob 0.2 --stroke-kernel-size 2
```

Artifact sẽ được lưu vào:

```text
artifacts/mnist_ann/
|-- best_model.pt
|-- metrics.json
|-- class_accuracy.json
`-- confusion_matrix.npy
```

Chạy smoke test nhanh:

```powershell
python scripts\train_mnist_ann.py --epochs 1 --limit-train 512 --limit-test 128 --batch-size 128 --cache-dir data\debug_cache --output-dir artifacts\debug_run
```

## Biên dịch báo cáo

Từ thư mục `report/` chạy:

```powershell
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
biber --input-directory build --output-directory build main
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
```

VS Code đã có sẵn recipe tương ứng trong `.vscode/settings.json`.
