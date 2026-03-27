# Colab Run For FashionMNIST

## Mục tiêu

Chạy cấu hình `single-branch ANN` trên `FashionMNIST` bằng GPU `T4` trên Google Colab.

Script dùng:

```powershell
python scripts\train_mnist_ann_single_branch.py --dataset fashionmnist
```

Script này sẽ tự tải `FashionMNIST` nếu trên Colab chưa có dữ liệu.

## Khuyến nghị Colab

- Runtime: `GPU`
- GPU khuyến nghị: `T4`
- Python: mặc định của Colab là đủ

## Thư viện cần có trên Colab

Chạy cell này trước:

```python
!pip install -q scikit-image opencv-python-headless
```

`torch` và `torchvision` thường đã có sẵn trên Colab.

## Cách dùng với VS Code + Colab extension

1. Mở repo này trong VS Code.
2. Kết nối notebook kernel tới Colab.
3. Đảm bảo thư mục repo đã có mặt trong Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Notebook hiện đã tự dò repo `Lab3-LT` trong Google Drive, nên bạn không cần `%cd` tay nữa. Nếu muốn tự kiểm tra path, có thể dùng:

```python
from pathlib import Path
list(Path('/content/drive/MyDrive').rglob('train_mnist_ann_single_branch.py'))[:5]
```

5. Kiểm tra GPU:

```python
!nvidia-smi
```

6. Chạy smoke test trước:

```python
!python scripts/train_mnist_ann_single_branch.py --dataset fashionmnist --epochs 1 --limit-train 1024 --limit-test 256 --batch-size 256 --output-dir artifacts/fashion_mnist_smoke
```

7. Nếu smoke test ổn, chạy full:

```python
!python scripts/train_mnist_ann_single_branch.py --dataset fashionmnist --epochs 50 --batch-size 256 --output-dir artifacts/fashion_mnist_ann_single_branch
```

8. Nếu muốn chạy cấu hình augmentation cục bộ:

```python
!python scripts/train_mnist_ann_single_branch.py --dataset fashionmnist --epochs 50 --batch-size 256 --cache-dir data/fashion_mnist_ann_single_branch_aug_cache --output-dir artifacts/fashion_mnist_ann_single_branch_augmented --max-rotate-deg 4 --max-shift-ratio 0.03 --max-shear-deg 3 --elastic-prob 0.45 --elastic-alpha 1.4 --elastic-sigma 4.5 --stroke-jitter-prob 0.2 --stroke-kernel-size 2
```

## Lưu kết quả

Notebook mặc định lưu kết quả vào chính repo trên Google Drive:

- `artifacts/fashion_mnist_ann_single_branch/`
- `artifacts/fashion_mnist_ann_single_branch_augmented/`

Vì repo của bạn đang sync với local, các file này sẽ xuất hiện lại ở máy local sau khi Colab ghi xong.

## Kết quả cần xem

Sau khi chạy xong, đọc:

- `artifacts/fashion_mnist_ann_single_branch/metrics.json`
- hoặc `artifacts/fashion_mnist_ann_single_branch_augmented/metrics.json`

Trường cần nhìn là:

- `test_acc`
- `final_train_acc`
- `generalization_gap`

## Gợi ý workflow

- Chạy `smoke test` trước để tránh đốt thời gian T4 nếu path hoặc package sai.
- Sau đó chạy `single-branch` full trước vì đây là cấu hình tốt nhất hiện tại trên MNIST.
- Nếu muốn tối ưu thêm, mới chạy biến thể có `elastic distortion + stroke jitter`.
