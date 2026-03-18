#!/bin/bash

python scripts/test.py \
--project yolov3_rtts_10c_deyolo \
--name deyolo \
--model deyolo.DEYOLO \
--img_size_test 544 544 \
--batch_size 8 \
--data ./configs/rtts_5c.yaml \
--hyp ./configs/hyp.voc.scratch.yaml \
--verbose \
--nms_thres 0.5 \
--conf_thres 0.001 \
--checkpoint pretrained_models/deyolo_foggy/best.pt
