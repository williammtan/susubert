#!/bin/bash

PIPELINE_NAME=$1"_pipeline" # eg. match, train

python $PIPELINE_NAME".py" compile
gsutil cp $PIPELINE_NAME".yaml" "gs://foodid_product_matching/pipelines/"$PIPELINE_NAME".yaml"

