FROM tensorflow/tensorflow:latest
WORKDIR /pipeline
COPY requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY train.py /pipeline