# Use a base image with Python
FROM nvcr.io/nvidia/tritonserver:24.07-py3-sdk

# Install necessary packages to add Docker repository and software-properties
RUN apt-get update && apt-get install -y \
    apt-utils ca-certificates curl gnupg lsb-release software-properties-common

# Add Docker’s official GPG key
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
RUN chmod a+r /etc/apt/keyrings/docker.asc

# Set up the stable repository
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 

# FROM docker:24.0-dind

# Set working directory
WORKDIR /dli

RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y build-essential bash curl unzip wget git libgl1-mesa-glx libglib2.0-0
RUN apt-get update && apt-get install -y python3.11 python3-pip 
# RUN apt-get update && apt-get install -y nvidia-container-toolkit nvidia-cuda-toolkit


#Run pip dependencies
COPY composer/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install ipywidgets jupyterlab==4.2.5 jupyter-archive==3.4.0 jupyter-resource-usage
COPY composer/jupyter/ /root/.jupyter
ENV SHELL /bin/bash
COPY composer/jupyter/bashrc /root/.bashrc

# COPY notebooks
WORKDIR /dli/task
# COPY . .

# Expose port 8888 for JupyterLab
EXPOSE 8888 
EXPOSE 9012

# Start JupyterLab when the container runs
ADD composer/entrypoint.sh /usr/local/bin
ENTRYPOINT [ "/usr/local/bin/entrypoint.sh" ]
