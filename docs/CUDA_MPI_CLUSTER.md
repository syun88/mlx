# CUDA + MPI Distributed Guide (Apple Silicon + NVIDIA)

## English

This note explains how to launch MLX distributed runs that pair an Apple Silicon host (M1/M2/M4) with an NVIDIA PC (e.g., RTX 3080) using the MPI backend instead of NCCL.

### Prerequisites
- Open MPI built with CUDA-awareness on **all** hosts. On macOS/Homebrew: `brew install open-mpi`; on Linux: build with `--with-cuda`.
- MLX built with CUDA and MPI enabled (`MLX_BUILD_CUDA=ON`; Open MPI available at build time). If the MPI library is in a non-standard path, set `MLX_MPI_LIBNAME=/path/to/libmpi.{dylib,so}`.
- SSH connectivity between hosts and a shared Python environment or synchronized working copy.

### Building MLX from source (example)
```bash
# On both Apple and NVIDIA hosts
python -m venv .venv && source .venv/bin/activate
CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install -e ".[dev]"
```

### Launching a run
```bash
# On the Apple host (rank 0), listing both machines
mlx.launch -n 2 --hosts apple.local,nvidia.local examples/distributed_mpi.py \
  --backend mpi
```
- Inside Python, pick the MPI backend explicitly:
  ```python
  import mlx.core as mx
  world = mx.distributed.init(backend="mpi")
  x = mx.distributed.all_sum(mx.ones(10, device=mx.Device(mx.Device.gpu)))
  ```
- The MPI backend will use the CUDA stream automatically when the default device is GPU; pass a CPU stream/device if you need CPU transfers.

### Operational notes
- CUDA graph capture is disabled for MPI calls; launches pay a sync before MPI.
- Unsupported on CUDA+MPI: float16/bfloat16 reductions and complex max/min. Cast to float32 or run those ops on CPU.
- `sum_scatter` is not implemented in the MPI backend.

### Typical topology
- Apple Silicon laptop/desktop (rank 0) + NVIDIA Linux box (rank 1+) on the same LAN; both running Open MPI with CUDA support. Use the same MLX commit on all nodes.

---

## 日本語

Apple Silicon (M1/M2/M4) と NVIDIA PC (例: RTX 3080) を組み合わせて NCCL の代わりに MPI バックエンドで MLX の分散実行を行う手順です。

### 前提条件
- 全ホストで CUDA 対応の Open MPI を導入（macOS/Homebrew: `brew install open-mpi`、Linux では `--with-cuda` 付きでビルド）。
- MLX を CUDA + MPI 対応でビルドする（`MLX_BUILD_CUDA=ON`、ビルド時に Open MPI を検出）。MPI が標準パスに無い場合は `MLX_MPI_LIBNAME=/path/to/libmpi.{dylib,so}` を設定。
- ホスト間の SSH 接続と、共通の Python 環境または同期された作業コピー。

### MLX のビルド例
```bash
# Apple / NVIDIA 両方で実施
python -m venv .venv && source .venv/bin/activate
CMAKE_ARGS="-DMLX_BUILD_CUDA=ON" pip install -e ".[dev]"
```

### 実行例
```bash
# Apple 側から (rank 0)、両ホストを指定
mlx.launch -n 2 --hosts apple.local,nvidia.local examples/distributed_mpi.py \
  --backend mpi
```
- Python 側では明示的に MPI バックエンドを選択:
  ```python
  import mlx.core as mx
  world = mx.distributed.init(backend="mpi")
  x = mx.distributed.all_sum(mx.ones(10, device=mx.Device(mx.Device.gpu)))
  ```
- デフォルトデバイスが GPU の場合は自動的に CUDA ストリームで MPI 通信が行われます。CPU で通信したい場合は CPU デバイス/ストリームを渡してください。

### 注意事項
- MPI 呼び出しでは CUDA グラフキャプチャを無効化。MPI 呼び出し前に CUDA ストリーム同期が入ります。
- CUDA+MPI では float16/bfloat16 の縮約、および complex の max/min は未対応。float32 へキャストするか CPU 実行に切り替えてください。
- `sum_scatter` は MPI バックエンドでは未実装です。

### 想定トポロジ
- 同一 LAN 上の Apple Silicon (rank 0) + NVIDIA Linux マシン (rank 1+)。両方で同じ MLX のコミットを使用してください。
