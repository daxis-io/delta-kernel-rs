#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0.

set -euo pipefail

manifest="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/Cargo.toml"
target="wasm32-unknown-unknown"

graph="$(
  cargo tree --manifest-path "$manifest" --target "$target" --locked \
    --prefix none -e normal,build
)"
feature_graph="$(
  cargo tree --manifest-path "$manifest" --target "$target" --locked \
    --prefix none -e normal,build,features
)"

denied_packages=(
  aws-lc-sys
  getrandom
  hyper
  liblzma-sys
  native-tls
  openssl-sys
  ring
  tempfile
  walkdir
  zstd-sys
)

for package in "${denied_packages[@]}"; do
  if grep -Eq "^${package} v" <<<"$graph"; then
    printf 'denied package in %s graph: %s\n' "$target" "$package" >&2
    grep -E "^${package} v" <<<"$graph" | LC_ALL=C sort -u >&2
    exit 1
  fi
done

denied_features=(
  'object_store feature "aws"'
  'object_store feature "azure"'
  'object_store feature "fs"'
  'object_store feature "gcp"'
  'tokio feature "rt-multi-thread"'
)

for feature in "${denied_features[@]}"; do
  if grep -Fq "$feature" <<<"$feature_graph"; then
    printf 'denied feature in %s graph: %s\n' "$target" "$feature" >&2
    exit 1
  fi
done

package_universes="$(
  cargo tree --manifest-path "$manifest" --target "$target" --locked \
    --prefix none -e normal,build --format '{p}' |
    sed 's/ (\*)$//' |
    LC_ALL=C sort -u
)"
for package in arrow parquet object_store; do
  universe_count="$(
    awk -v package="$package" '$1 == package { count++ } END { print count + 0 }' \
      <<<"$package_universes"
  )"
  if (( universe_count > 1 )); then
    printf 'duplicate %s source/version universe in %s graph\n' "$package" "$target" >&2
    awk -v package="$package" '$1 == package' <<<"$package_universes" >&2
    exit 1
  fi
done

printf 'Buoyant Kernel %s dependency policy passed\n' "$target"
