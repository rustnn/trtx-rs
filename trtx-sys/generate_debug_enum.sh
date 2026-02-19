#! /bin/sh
set -exu

cargo install bindgen-cli

echo "#![allow(warnings)]" > src/enums.rs

bindgen \
  --default-enum-style=rust \
  --with-derive-default \
  --with-derive-eq \
  --with-derive-hash \
  --with-derive-ord \
  --allowlist-type "(.*Type|.*Mode|.*Operation|.*Strategy|.*Severity|.*Format)" \
  --blocklist-type "(cu.*)" \
  --generate-inline-functions \
  ./TensorRT-Headers/TRT-RTX-1.3/NvInfer.h \
  -- -x c++ \
  >> src/enums.rs

sed -i 's/extern "C"/extern "system"/g' src/enums.rs
sed -i 's/nvinfer1_//g' src/enums.rs
sed -i 's/"!/"/g' src/enums.rs
sed -i 's/\\n!/\\n/g' src/enums.rs
cargo fmt
