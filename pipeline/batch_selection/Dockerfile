FROM gcr.io/food-id-app/susubert/feature_extraction@sha256:86821274ebf5c161cde58a5aa20f23b953b4848bc5e10aefde36f6a4c7d2abe5
WORKDIR /pipeline
COPY requirements.txt /pipeline
RUN pip install -r requirements.txt
COPY batch_selection.py /pipeline
COPY oneshot.py /pipeline