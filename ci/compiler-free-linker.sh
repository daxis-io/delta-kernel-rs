#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0.

set -euo pipefail

readonly system_lib=/usr/lib/x86_64-linux-gnu
readonly gcc_lib=/usr/lib/gcc/x86_64-linux-gnu/12
readonly dynamic_linker=/lib64/ld-linux-x86-64.so.2

for object in \
  "$system_lib/Scrt1.o" \
  "$system_lib/crti.o" \
  "$gcc_lib/crtbeginS.o" \
  "$gcc_lib/crtendS.o" \
  "$system_lib/crtn.o"
do
  test -f "$object"
done

linker_args=()
shared_output=false
for argument in "$@"; do
  case "$argument" in
    -shared)
      shared_output=true
      linker_args+=("$argument")
      ;;
    -m64)
      linker_args+=("-m" "elf_x86_64")
      ;;
    -nodefaultlibs)
      ;;
    -Wl,*)
      IFS=',' read -r -a forwarded <<<"${argument#-Wl,}"
      linker_args+=("${forwarded[@]}")
      ;;
    *)
      linker_args+=("$argument")
      ;;
  esac
done

startup_objects=(
  "$system_lib/crti.o"
  "$gcc_lib/crtbeginS.o"
)
linker_tail=(
  "$gcc_lib/crtendS.o"
  "$system_lib/crtn.o"
)
if [[ "$shared_output" == false ]]; then
  startup_objects=("$system_lib/Scrt1.o" "${startup_objects[@]}")
  linker_tail+=("--dynamic-linker=$dynamic_linker")
fi

exec ld.lld \
  "${startup_objects[@]}" \
  -L/lib/x86_64-linux-gnu \
  -L"$system_lib" \
  -L"$gcc_lib" \
  "${linker_args[@]}" \
  "${linker_tail[@]}"
