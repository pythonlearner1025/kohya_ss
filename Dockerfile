FROM nvcr.io/nvidia/pytorch:23.04-py3 as base
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt-get install -y git curl libgl1 libglib2.0-0 libgoogle-perftools-dev \
	                   python3.10 python3.10-tk python3-html5lib python3-apt python3-pip python3.10-distutils && \
	rm -rf /var/lib/apt/lists/*

# Set python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 3 && \
	update-alternatives --config python3

RUN useradd -m -s /bin/bash appuser
USER appuser

WORKDIR /app

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN python3 -m pip install wheel

# Install requirements
COPY requirements.txt setup.py .
RUN python3 -m pip install --use-pep517 -U -r requirements.txt

# Upgrade to Torch 2.0
RUN python3 -m pip install --use-pep517 --no-deps -U triton==2.0.0 torch>=2.0.0+cu121 xformers==0.0.17 \
	                       --extra-index-url https://download.pytorch.org/whl/cu121

# Fix missing libnvinfer7
USER root
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

USER appuser
COPY --chown=appuser . .

# https://github.com/kohya-ss/sd-scripts/issues/405#issuecomment-1509851709
RUN sed -i 's/import library.huggingface_util/# import library.huggingface_util/g' train_network.py && \
    sed -i 's/import library.huggingface_util/# import library.huggingface_util/g' library/train_util.py

STOPSIGNAL SIGINT
ENV LD_PRELOAD=libtcmalloc.so
CMD python3 "./kohya_gui.py" ${CLI_ARGS} --listen 0.0.0.0 --server_port 7860