#!/bin/bash
# COLMAP 3.14-dev from the GPU glomap build (the /usr/local/bin one needs libcudart.13 and is broken).
export LD_LIBRARY_PATH=/home/paperspace/code/_cuda12/lib:/home/paperspace/code/_deps/libcudss-linux-x86_64-0.4.0.2_cuda12-archive/lib:/home/paperspace/code/_deps/absl-install/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
exec /home/paperspace/code/glomap/build_gpu/_deps/colmap-build/src/colmap/exe/colmap "$@"
