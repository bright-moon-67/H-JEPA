#!/usr/bin/env bash
# Parallel ranged download of the Push-T expert-train archive.
# Single-connection throughput to hf-mirror is ~3-5 MB/s; 24 parallel ranges
# aggregate considerably more. Parts are resumable (curl -C -), then joined.
set -u
URL="${URL_OVERRIDE:-https://hf-mirror.com/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst}"
SIZE=13136247974
N=${N:-48}          # number of ranges
P=${P:-24}          # parallel workers
OUT_DIR=${OUT_DIR:-/root/rivermind-data/hjepa_home/train_parts_n${N}}
OUT_FILE=${OUT_FILE:-/root/rivermind-data/hjepa_home/pusht_expert_train.h5.zst}

mkdir -p "$OUT_DIR"
CH=$(( (SIZE + N - 1) / N ))

fetch_one() {
  local i=$1
  local s=$(( i * CH ))
  local e=$(( s + CH - 1 ))
  [ "$e" -ge "$SIZE" ] && e=$(( SIZE - 1 ))
  local want=$(( e - s + 1 ))
  local part="$OUT_DIR/part.$(printf '%04d' "$i")"
  if [ -f "$part" ]; then
    local have
    have=$(stat -c%s "$part" 2>/dev/null || echo 0)
    # Strict equality: a part left over from a DIFFERENT split count has the
    # same name but a different length, and silently reusing it corrupts the
    # concatenated archive.
    if [ "$have" -eq "$want" ]; then return 0; fi
    rm -f "$part"
  fi
  curl -sL --retry 5 --retry-delay 2 --speed-time 60 --speed-limit 1024 \
    -C - -r "$s-$e" -o "$part" "$URL"
  local have
  have=$(stat -c%s "$part" 2>/dev/null || echo 0)
  if [ "$have" -ne "$want" ]; then
    echo "INCOMPLETE $i ($have/$want)"
    rm -f "$part"
  fi
  return 0
}
export -f fetch_one
export URL SIZE N CH OUT_DIR

seq 0 $((N-1)) | xargs -P "$P" -I@ bash -c 'fetch_one @'

# verify then join
total=0
for i in $(seq 0 $((N-1))); do
  part="$OUT_DIR/part.$(printf '%04d' "$i")"
  sz=$(stat -c%s "$part" 2>/dev/null || echo 0)
  total=$(( total + sz ))
done
echo "downloaded bytes: $total / $SIZE"
if [ "$total" -ne "$SIZE" ]; then
  echo "SIZE MISMATCH - rerun to resume"
  exit 1
fi

: > "$OUT_FILE"
for i in $(seq 0 $((N-1))); do
  cat "$OUT_DIR/part.$(printf '%04d' "$i")" >> "$OUT_FILE"
done
echo "joined -> $OUT_FILE"
rm -rf "$OUT_DIR"
