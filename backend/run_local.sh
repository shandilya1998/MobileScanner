#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts performs cloud training for a Tensorflow model.
echo "Training cloud ML model"

# IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry
IMAGE_REPO_NAME=mobilescanner

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=object_detection

PROJECT_ID=mobilescanner-297614

BUCKET_ID=mobilescanner_bucket

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=${IMAGE_REPO_NAME}:${IMAGE_TAG}

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=custom_gpu_container_job_$(date +%Y%m%d_%H%M%S)

# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the model will be deployed.
REGION=us-central1

# Build the docker image
docker build -t ${IMAGE_URI} ./

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_ID}/models/gpu
# Note: these files have already been copied over when the image was built

docker run -it ${IMAGE_URI} \
    --job-dir ${JOB_DIR} \
    --train-batch-size 10 \
    --val-batch-size 10 \
    --learning-rate 0.005 \
    --eps 10e-8 \
    --beta1 0.9 \
    --beta2 0.999 \
    --epochs 600 \
    --seed 42 \
    --net yolov2 \
    --log-interval 10 \

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/checkpoint
gsutil ls ${JOB_DIR}/test
gsutil ls ${JOB_DIR}/logs
