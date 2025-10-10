FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
COPY environment.yml /tmp/environment.yml

# setup the web proxy for Internet access

# configure the ubuntu's mirror
RUN apt-get update
RUN apt-get install -y wget git build-essential zip unzip vim


# install Miniconda (or Anaconda)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-0-Linux-x86_64.sh -O miniconda.sh \
    && /bin/bash miniconda.sh -b -p /softwares/miniconda3 \
    && rm -v miniconda.sh
ENV PATH="/softwares/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/softwares/miniconda3/lib:${LD_LIBRARY_PATH}"

# install Python packages
RUN pip install --upgrade pip

# Override CUDA detection for conda
ENV CONDA_OVERRIDE_CUDA=12.6

# Accept channels Terms of Service
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# update conda
# RUN conda config --set ssl_verify false
RUN conda update -n base conda

RUN conda config --add channels rmg
RUN conda config --add channels conda-forge
RUN conda config --add channels rdkit
RUN conda config --add channels pytorch
RUN conda update -n base xz
RUN conda env create -n kermt -f /tmp/environment.yml

# clean-up
RUN rm -rf /var/lib/apt/lists/*
# Clean up older xz package
RUN rm -rf /softwares/miniconda3/pkgs/xz-5.6.4-h5eee18b_1/
# RUN apt clean && apt autoremove -y

# COPY code and cleanup
COPY . /code
RUN rm -rf /code/.git /code/.code-workspace

# Equivalent to `conda activate kermt`
SHELL ["conda", "run", "--no-capture-output", "-n", "kermt", "/bin/bash", "-c"]

# Install the cuik_molmaker from wheel
RUN pip install cuik_molmaker==0.1.1 --index-url https://pypi.nvidia.com/rdkit-2025.03.2_torch-2.7.1/

# provide defaults for the executing container
CMD [ "/bin/bash" ]