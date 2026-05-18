#!/usr/bin/env bash
# V5 helper: 下载 MicroLens-100K 官方 5 frames 分卷包。
# 用法：
#   bash scripts/v5/download_microlens_100k_frames.sh
# 下载后请在同一目录解压主 .zip，再把解压出的图片目录路径写入 configs/v5/profile_generation.yaml 的 data.frames_dir。

set -euo pipefail

BASE_URL="${1:-https://recsys.westlake.edu.cn/MicroLens-100k-Dataset}"
OUT_DIR="${2:-data/raw/microlens_100k/frame_archives}"
PREFIX="MicroLens-100k_frames_interval_1_number_5"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

build_url_file() {
  local url_file="frames_urls.txt"
  : > "$url_file"
  printf "%s/%s.zip\n" "${BASE_URL%/}" "$PREFIX" >> "$url_file"
  for i in $(seq 1 8); do
    printf "%s/%s.z%02d\n" "${BASE_URL%/}" "$PREFIX" "$i" >> "$url_file"
  done
  echo "$url_file"
}

if command -v aria2c >/dev/null 2>&1; then
  url_file="$(build_url_file)"
  echo "[INFO] aria2c detected. Downloading split archives as independent files."
  aria2c \
    -c \
    -x 4 \
    -s 4 \
    -j 2 \
    -k 1M \
    --max-tries=0 \
    --retry-wait=30 \
    --connect-timeout=30 \
    --timeout=60 \
    --summary-interval=30 \
    --auto-file-renaming=false \
    --file-allocation=none \
    --log=aria2_frames.log \
    --log-level=notice \
    -i "$url_file"
  echo "[DONE] Downloaded ${PREFIX} archives into $OUT_DIR"
  exit 0
fi

download_one() {
  local name="$1"
  local url="${BASE_URL%/}/${name}"
  if wget --spider -q "$url"; then
    echo "[DOWNLOADING] $name"
    wget -c "$url" -O "$name"
    return 0
  fi
  return 1
}

download_one "${PREFIX}.zip"

i=1
while true; do
  part="$(printf "%s.z%02d" "$PREFIX" "$i")"
  if ! download_one "$part"; then
    break
  fi
  i=$((i + 1))
done

echo "[DONE] Downloaded ${PREFIX} archives into $OUT_DIR"
