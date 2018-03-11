#!/bin/bash

COCO_VAL=http://images.cocodataset.org/zips/val2014.zip
COCO_ANNOTATION=http://images.cocodataset.org/annotations/annotations_trainval2014.zip

mkdir ../data/coco
mkdir ../data/coco/images

echo "Downloading coco val images ..."
wget $COCO_VAL -O val2014.zip
echo "Unzipping..."
7z x val2014.zip -o../data/coco/images

echo "Downloading coco annotation data ..."
wget $COCO_ANNOTATION -O annotations_trainval2014.zip
echo "Unzipping..."
unzip -q annotations_trainval2014.zip -d ../data/coco

rm val2014.zip annotations_trainval2014.zip 
