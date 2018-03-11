#!/bin/bash

COCO_AUX_URL=www.cs.virginia.edu/~ft3ex/data/who_where/coco_aux.zip
PRTRAINED_URL=www.cs.virginia.edu/~ft3ex/data/who_where/pretrained_model_and_testset.zip

echo "Downloading auxiliary data for coco ..."
wget $COCO_AUX_URL -O coco_aux.zip
echo "Unzipping..."
unzip -q coco_aux.zip -d ../data

echo "Downloading pretrained model and testset ..."
wget $PRTRAINED_URL -O pretrained_model_and_testset.zip
echo "Unzipping..."
unzip -q pretrained_model_and_testset.zip -d ../data

rm coco_aux.zip pretrained_model_and_testset.zip
