FROM tensorflow/tensorflow:latest-gpu
WORKDIR /pipeline
COPY requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY batch_selection.py /pipeline
COPY oneshot.py /pipeline