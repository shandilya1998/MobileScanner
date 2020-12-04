FROM tensorflow/tensorflow:2.1.0-gpu-py3

WORKDIR /root

# Copies all the relevant files to the docker image
COPY ./src/object_detection/train.py ./src/object_detection/train.py
COPY ./src/object_detection/tiny_yolo.py ./src/object_detection/tiny_yolo.py
COPY ./src/object_detection/yolo_lite.py ./src/object_detection/yolo_lite.py
COPY ./src/object_detection/yolov2.py ./src/object_detection/yolov2.py
COPY ./src/object_detection/loss.py ./src/object_detection/loss.py
COPY ./src/object_detection/constants.py ./src/object_detection/constants.py
COPY ./src/object_detection/mobilenetv2.py ./src/object_detection/mobilenetv2.py
COPY ./src/object_detection/gnd_truth.py ./src/object_detection/gnd_truth.py

ADD ./data/object_detection/images ./data/object_detection/images
ADD ./data/object_detection/annotations ./data/object_detection/annotations

RUN pip3 install --upgrade pip

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "-u" "run.py"]
