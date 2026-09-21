#!/bin/bash
# Build the hierarchical-3d-gaussians conda env on the A100 box.
# Toolchain: python 3.12, torch 2.4.1+cu124 (matches the CUDA 12.4 toolkit at ~/code/_cuda12 used for the GPU glomap build).
set -o pipefail
export CUDA_HOME=/home/paperspace/code/_cuda12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=8
source /home/paperspace/miniconda3/etc/profile.d/conda.sh
cd /home/paperspace/code/hierarchical-3d-gaussians
conda env list | grep -q '^h3dgs ' || conda create -n h3dgs python=3.12 -y
conda activate h3dgs
python -c "import torch" 2>/dev/null || pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install plyfile tqdm joblib exif scikit-learn opencv-python matplotlib pillow
pip install --no-build-isolation ./submodules/hierarchy-rasterizer 2>&1 | tail -3
pip install --no-build-isolation ./submodules/simple-knn 2>&1 | tail -3
pip install --no-build-isolation ./submodules/gaussianhierarchy 2>&1 | tail -3
python -c "import torch, diff_gaussian_rasterization, simple_knn; print('py ext OK torch', torch.__version__, torch.version.cuda, 'cuda', torch.cuda.is_available())"
(cd submodules/gaussianhierarchy && cmake . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3 && cmake --build build -j 8 --config Release 2>&1 | tail -3)
ls -la submodules/gaussianhierarchy/build/GaussianHierarchyCreator submodules/gaussianhierarchy/build/GaussianHierarchyMerger
echo "H3DGS ENV DONE $(date)"
