FROM tensorflow/tensorflow:2.5.0-gpu
WORKDIR /pipeline
COPY requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY train.py /pipeline