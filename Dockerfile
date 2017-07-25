# Base image
FROM tensorflow/tensorflow:latest-gpu

# Install extra dependencies
RUN apt-get update && \
    apt-get -y install git && \
    apt-get -y install vim && \
    apt-get -y install unzip

#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

RUN pip --no-cache-dir install keras

#RUN pip install -r requirements.txt
RUN git clone https://github.com/ifigueroa/deep-learning-examples.git /deep-learning-examples
CMD exec /bin/bash -c "trap : TERM INT; sleep infinty & wait"
