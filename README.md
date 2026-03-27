# Lab3-LT

Repository này chứa toàn bộ mã nguồn, notebook và báo cáo cho Lab 03 môn Thị giác máy tính. Trọng tâm của bài là xây dựng và đánh giá các biến thể ANN trên ba nhóm dữ liệu:

- `MNIST`
- `FashionMNIST`
- `ChestMNIST` trong họ `MedMNIST`

## Cấu trúc repo

```text
.
|-- notebooks/
|   |-- fashionmnist/
|   `-- chestmnist/
|-- report/
|   |-- assets/
|   |-- figures/
|   |-- sections/
|   |-- tables/
|   `-- main.tex
|-- scripts/
|-- data/          # dữ liệu local, không track git
|-- artifacts/     # kết quả train local, không track git
|-- COLAB.md
`-- README.md
```

## Ý nghĩa các thư mục chính

- `scripts/`: script train, evaluate và generate diagnostics.
- `notebooks/`: notebook Colab đã chuẩn bị sẵn theo từng dataset.
- `report/`: toàn bộ source LaTeX của báo cáo.
- `data/`: cache và dataset local, chỉ dùng khi chạy thực nghiệm.
- `artifacts/`: checkpoint, metrics và output sau khi train.

## Notebook Colab

`FashionMNIST`

- `notebooks/fashionmnist/fashion_mnist_naive_colab.ipynb`
- `notebooks/fashionmnist/fashion_mnist_colab.ipynb`
- `notebooks/fashionmnist/fashion_mnist_two_branch_colab.ipynb`
- `notebooks/fashionmnist/fashion_mnist_single_branch_ensemble_colab.ipynb`
- `notebooks/fashionmnist/fashion_mnist_single_branch_ensemble_eval_colab.ipynb`
- `notebooks/fashionmnist/fashion_mnist_region_fusion_ensemble_colab.ipynb`

`ChestMNIST`

- `notebooks/chestmnist/chestmnist_naive_single_branch_colab.ipynb`
- `notebooks/chestmnist/chestmnist_cv_weighted_ann_colab.ipynb`
- `notebooks/chestmnist/chestmnist_feature_fusion_ann_colab.ipynb`
- `notebooks/chestmnist/chestmnist_feature_fusion_ann_lite_colab.ipynb`

## Chạy nhanh bằng script

`MNIST naive`

```powershell
python scripts\train_mnist_ann_naive.py --epochs 20 --batch-size 256
```

`MNIST single-branch`

```powershell
python scripts\train_mnist_ann_single_branch.py --epochs 20 --batch-size 256
```

`MNIST two-branch`

```powershell
python scripts\train_mnist_ann.py --epochs 20 --batch-size 256
```

`FashionMNIST single-branch`

```powershell
python scripts\train_mnist_ann_single_branch.py --dataset fashionmnist --epochs 50 --batch-size 256
```

`FashionMNIST two-branch`

```powershell
python scripts\train_mnist_ann.py --dataset fashionmnist --epochs 50 --batch-size 256
```

`FashionMNIST naive`

```powershell
python scripts\train_fashion_ann_naive.py --epochs 50 --batch-size 256
```

## Biên dịch báo cáo

Từ thư mục `report/` chạy:

```powershell
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
biber --input-directory build --output-directory build main
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
xelatex -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=build main.tex
```

PDF đầu ra nằm ở:

- `report/build/main.pdf`

## Ghi chú

- `data/` và `artifacts/` được giữ local để repo gọn, không đẩy lên GitHub.
- Toàn bộ link GitHub của project đã được chèn trực tiếp vào báo cáo.
- Hướng dẫn chạy Colab tổng quát nằm trong `COLAB.md`.
