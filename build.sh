#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ARCH=${1:-native}
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -eq 0 ]; then set -- glu-original c94 c104 c106; fi
for tool in nvcc gcc g++ make; do command -v "$tool" >/dev/null; done
for version in "$@"; do
    case "$version" in glu-original|c94|c104|c106) ;; *) echo "Unknown version: $version" >&2; exit 1;; esac
    SRC="$ROOT/versions/$version/src"
    echo "Building $version for $ARCH"
    make -C "$SRC/nicslu/lib" clean all
    make -C "$SRC/nicslu/util" clean all
    make -C "$SRC" clean
    make -C "$SRC" MAIN NVCC=nvcc \
        NVCCFLAGS="-O3 -lineinfo -std=c++11 -arch=$ARCH -Ipreprocess -I../include -Inicslu/include"
done
