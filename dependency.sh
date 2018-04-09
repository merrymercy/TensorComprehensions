##### Tensor Comprehension #####
 sudo apt-get install -y libgoogle-glog-dev curl build-essential cmake git automake libgmp3-dev libtool ssh libyaml-dev realpath wget valgrind software-properties-common unzip libz-dev
 
# tapir llvm
export CC=$(which gcc)
export CXX=$(which g++)
export CORES=$(nproc)
export LLVM_SOURCES=$HOME/llvm_sources-tapir5.0
export CLANG_PREFIX=$HOME/clang+llvm-tapir5.0  # change this to whatever path you want
export CMAKE_VERSION=cmake
mkdir -p $LLVM_SOURCES && cd $LLVM_SOURCES

git clone --recursive https://github.com/wsmoses/Tapir-LLVM llvm
mkdir -p ${LLVM_SOURCES}/llvm_build && cd ${LLVM_SOURCES}/llvm_build
${CMAKE_VERSION} -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_OCAMLDOC=OFF -DLLVM_INSTALL_OCAMLDOC_HTML_DIR=/tmp -DLLVM_OCAML_INSTALL_PATH=/tmp -DCMAKE_INSTALL_PREFIX=${CLANG_PREFIX} -DLLVM_TARGETS_TO_BUILD=X86 -DCOMPILER_RT_BUILD_CILKTOOLS=OFF -DLLVM_ENABLE_CXX1Y=ON -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_BUILD_TESTS=OFF -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release -DLLVM_BUILD_LLVM_DYLIB=ON  -DLLVM_ENABLE_RTTI=ON ../llvm/
make -j $CORES -s && sudo make install -j $CORES -s
cd $HOME

# protobuf
mkdir -p /tmp/proto-install && cd /tmp/proto-install
wget --quiet https://github.com/google/protobuf/archive/v3.4.0.zip -O proto.zip && unzip -qq proto.zip -d .
cd protobuf-3.4.0 && ./autogen.sh && ./configure && make -j$N_CORE && sudo make install && ldconfig

$PIP pyyaml

cd $HOME && git clone https://github.com/facebookresearch/TensorComprehensions.git --recursive
cd TensorComprehensions
git submodule update --init --recursive
export TC_DIR=$(pwd)
BUILD_TYPE=Release PYTHON=$(which python3) WITH_CAFFE2=OFF CLANG_PREFIX=$HOME/clang+llvm-tapir5.0 ./build.sh --all
