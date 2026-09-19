#!/bin/bash
# Build FFmpeg 7.1 with shared libraries for torchcodec 0.6.0 (needs libavutil.so.59).
# Run INSIDE the container, e.g.:
#   module load singularity
#   singularity exec -e --nv /app1/common/singularity-img/hopper/pytorch/pytorch_2.6.0_cuda_12.8.sif \
#     bash scripts/lingbot/build_ffmpeg7.sh
set -euo pipefail

VERSION=7.1
PREFIX=/scratch/e1583535/opt/ffmpeg-${VERSION}
BUILD_DIR=/scratch/e1583535/tmp/ffmpeg-build-${VERSION}

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -f "ffmpeg-${VERSION}.tar.xz" ]; then
    curl -fL -o "ffmpeg-${VERSION}.tar.xz" "https://ffmpeg.org/releases/ffmpeg-${VERSION}.tar.xz"
fi

rm -rf "ffmpeg-${VERSION}"
tar xf "ffmpeg-${VERSION}.tar.xz"
cd "ffmpeg-${VERSION}"

# x86 assembly needs nasm/yasm; disable it if neither is present (slower but builds cleanly).
ASM_FLAG=""
if ! command -v nasm >/dev/null 2>&1 && ! command -v yasm >/dev/null 2>&1; then
    ASM_FLAG="--disable-x86asm"
fi

./configure \
    --prefix="$PREFIX" \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --disable-programs \
    $ASM_FLAG

make -j"$(nproc)"
make install

echo "==================================================="
echo "FFmpeg ${VERSION} installed to $PREFIX"
ls -1 "$PREFIX/lib/"libavutil.so.* 2>/dev/null
echo "Add to LD_LIBRARY_PATH: $PREFIX/lib"
echo "==================================================="
