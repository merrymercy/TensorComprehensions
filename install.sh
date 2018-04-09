sudo cp /usr/bin/python backup
sudo cp /usr/bin/python3 /usr/bin/python

git submodule update --init --recursive
export TC_DIR=$(pwd)
BUILD_TYPE=Release PYTHON=$(which python3) WITH_CAFFE2=OFF CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all

sudo mv backup /usr/bin/python
