USE_CUDA=$1
if [ -z "$USE_CUDA" ]; then
    echo "Usage: $0 <USE_CUDA>"
    echo "USE_CUDA: 0 for CPU only, 1 for GPU"
    exit 1
fi

git clone https://github.com/colmap/colmap.git /root/colmap
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev \
    libopengl0 \
    mesa-common-dev \
    libglu1-mesa-dev \
    freeglut3-dev

cd /root/colmap/
rm -rf build
mkdir build
cd build

# Enable/Disable CUDA support
if [ "$USE_CUDA" -eq 0 ]; then
    echo "CUDA support disabled"
    cmake .. -GNinja -DCUDA_ENABLED=OFF
else
    echo "CUDA support enabled"
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=80
fi

ninja
ninja install
