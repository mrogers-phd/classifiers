# Docker to generate containers for assembly tests.
FROM ubuntu

# Eliminate LANG questions
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install bash python3-dev python3-pip 
RUN apt install python-is-python3

RUN pip3 install --upgrade pip

# Enable Jupyter
RUN pip3 install matplotlib scikit-learn jupyter

# Copy local scripts and modules
COPY shared/ /home/classifiers/shared/
COPY bin/ /home/classifiers/bin/
COPY data/ /home/classifiers/data/
COPY README.md /home/classifiers/README.md

ENV HOME=/home/classifiers
ENV PYTHONPATH=.:$HOME/bin:$HOME/shared
ENV PATH=.:$PATH:$HOME/bin
ENV JUPYTER_PATH=$PYTHONPATH

WORKDIR $HOME

# Jupyter notebooks
COPY *.ipynb $HOME/

RUN chmod a+rx $HOME/*.*

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0",  "--allow-root"]
