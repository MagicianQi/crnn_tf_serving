#!/usr/bin/env bash

sudo docker run \
--runtime=nvidia \
--name text_recognition \
--restart always \
-e CUDA_VISIBLE_DEVICES=2 \
-p 8500:8500 -p 8501:8501 \
--mount type=bind,source=/home/qishuo/code/crnn_tf_serving/models/crnn,target=/models/crnn \
-t --entrypoint=tensorflow_model_server tensorflow/serving:1.13.0-gpu \
--port=8500 --rest_api_port=8501 \
--model_name=crnn \
--model_base_path=/models/crnn \
--enable_batching \
--per_process_gpu_memory_fraction=0.4
