git submodule update --init --recursive
export TC_DIR=$(pwd)
BUILD_TYPE=Release PYTHON=$(which python3) WITH_CAFFE2=OFF CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all
