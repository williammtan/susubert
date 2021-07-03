FROM tensorflow/tensorflow:latest-gpu
WORKDIR /pipeline
COPY requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY feature_extraction.py /pipeline