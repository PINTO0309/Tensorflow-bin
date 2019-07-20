#!/bin/bash
TARGETS=(
  "*.whl"
)

target=$(printf " %s" "${TARGETS[@]}")
target=${target:1}

git filter-branch --index-filter "git rm -r --cached --ignore-unmatch ${target}" -- --all

