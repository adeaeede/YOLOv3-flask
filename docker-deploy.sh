#!/usr/bin/env bash
$(aws ecr get-login --no-include-email --region eu-central-1)
docker build -t yolov3-object-detection .
docker tag yolov3-object-detection:latest 315222543498.dkr.ecr.eu-central-1.amazonaws.com/yolov3-object-detection:latest
docker push 315222543498.dkr.ecr.eu-central-1.amazonaws.com/yolov3-object-detection:latest
