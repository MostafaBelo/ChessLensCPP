# ChessLensCPP

## Installation

```
sudo apt install libopencv-dev cmake libcurl4-openssl-dev

wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz && tar -xzf onnxruntime-linux-aarch64-1.23.2.tgz && rm onnxruntime-linux-aarch64-1.23.2.tgz && mv onnxruntime-linux-aarch64-1.23.2 ~/onnxruntime

export LD_LIBRARY_PATH=~/onnxruntime/lib:$LD_LIBRARY_PATH

mkdir build
cd build

cmake .. \
    -DONNXRUNTIME_INCLUDE_DIR=~/onnxruntime/include \
    -DONNXRUNTIME_LIB_DIR=~/onnxruntime/lib \
    -DUSE_HTTP=ON

cd ..
```

## Build & Run

```
cmake --build ./build -j$(nproc) && ./build/chesslens_main
```
