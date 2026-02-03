# ChessLensCPP

## Installation

```
sudo apt install libopencv-dev cmake libcurl4-openssl-dev

wget https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz && tar -xzf onnxruntime-linux-x64-1.23.2.tgz && rm onnxruntime-linux-x64-1.23.2.tgz && mv onnxruntime-linux-x64-1.23.2 ~/onnxruntime

mkdir build

cmake . \
    -DONNXRUNTIME_INCLUDE_DIR=~/onnxruntime/include \
    -DONNXRUNTIME_LIB_DIR=~/onnxruntime/lib \
    -DUSE_HTTP=ON
```

## Build & Run

```
cmake --build ./build -j$(nproc) && ./build/chesslens_main
```
