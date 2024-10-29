FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    cmake \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/vinthony/video-retalking.git /video-retalking

RUN conda create -n video_retalking python=3.8 -y && \
    echo "source activate video_retalking" > ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc && conda activate video_retalking && \
    conda install -y ffmpeg && \
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r /video-retalking/requirements.txt"

ENV CONDA_DEFAULT_ENV=video_retalking
ENV PATH /opt/conda/envs/video_retalking/bin:$PATH

WORKDIR /video-retalking
CMD ["/bin/bash"]
