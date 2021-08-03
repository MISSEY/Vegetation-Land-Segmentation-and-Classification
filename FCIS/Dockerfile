# FCIS supports ubuntu 14.04
FROM nvidia/cuda:8.0-devel-ubuntu14.04

# remove prebuild cuda dependencies if image has
RUN rm /etc/apt/sources.list.d/cuda.list

# install all the packages required for mxnet 0.10.1
RUN sudo apt-get update -y && apt-get install -y  locales lsb-release \
    build-essential \
    libreadline-gplv2-dev  \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    wget \
    libopenblas-dev \
    libopencv-dev \
    git

# Install python2.7 which is hard requirement for original fcis
RUN mkdir software && cd software && \
    wget https://www.python.org/ftp/python/2.7.9/Python-2.7.9.tgz && \
    tar -xvf Python-2.7.9.tgz && \
    cd Python-2.7.9 && \
    ./configure && \
    make && \
    sudo make install

# installation of pip
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py && python get-pip.py
RUN pip install Cython matplotlib opencv-python==3.2.0.6 pillow PyYAML scikit-image scipy easydict==1.6 hickle

# Below steps are for installation of FCIS and MXNET from official git repos
# https://github.com/msracver/FCIS
# https://github.com/dmlc/mxnet

# initialise FCIS
RUN git clone https://github.com/msracver/FCIS.git && \
    cd FCIS && \
    sh init.sh

# clone mxnet and checkout to 0.10.1 which is required by FCIS
RUN git clone --recursive https://github.com/dmlc/mxnet.git && \
    cd mxnet && \
    git checkout 998378a && \
    git submodule init && \
    git submodule update

RUN cp -r FCIS/fcis/operator_cxx/channel_operator* mxnet/src/operator/contrib/

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Sk_yecX2jgQ6g2dyQaTZqjHWltag8Umx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Sk_yecX2jgQ6g2dyQaTZqjHWltag8Umx" -O cudnn-8.0-linux-x64-v6.0-tgz && rm -rf /tmp/cookies.txt

# Setup cuda 8.0 and cudnn
RUN tar zxvf cudnn-8.0-linux-x64-v6.0-tgz && \
    sudo cp cuda/include/cudnn*.h /usr/local/cuda/include && \
    sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

# install mxnet supporting cudnn and opencv
RUN cd mxnet && \
    make -j $(nproc) USE_OPENCV=1 USE_BLAS=openblas USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda USE_CUDNN=1 && \
    cd python && \
    sudo python setup.py install
