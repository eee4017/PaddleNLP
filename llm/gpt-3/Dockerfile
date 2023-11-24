FROM nvcr.io/nvidia/paddlepaddle:23.10-py3

RUN cd /workspace && \
    git clone https://github.com/NVIDIA/TransformerEngine && \
    cd TransformerEngine && \
    git submodule update --init --recursive && \
    NVTE_FRAMEWORK=paddle pip install .

RUN cd /workspace && \
    git clone https://github.com/eee4017/PaddleNLP.git && \
    cd PaddleNLP && \
    git checkout sc_test && \
    pip install .

RUN pip install regex tool_helpers visualdl==2.5.3 wandb