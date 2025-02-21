FROM continuumio/anaconda3

COPY environment.yml workspace/DAM-TAG-DYM/environment.yml 

# Change shell to be able to easily activate virtualenv
SHELL ["/bin/bash", "-c"]
WORKDIR /workspace

RUN conda env create -f DAM-TAG-DYM/environment.yml

SHELL ["conda", "run", "-n", "DAM-TAG-DYM", "/bin/bash", "-c"]

RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip install torch_geometric

RUN umask 022 && apt-get update \
    # Install system packages
    && apt-get install -y --no-install-recommends apt-utils ca-certificates gosu sudo git \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT [ "/workspace/entrypoint.sh" ]