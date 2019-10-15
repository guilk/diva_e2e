#!/usr/bin/env bash
# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

### Generate Image From Video

### please change the corresponding path prefix ${PATH} 


avi=`find /mnt/sda/wenhe/THUMOS14/test/TH14_test_set_mp4 -name \*.mp4`

detection=`cat /mnt/sda/diva/thumos14/annotations/TH14_Temporal_annotations_test/TH14_Temporal_Annotations_Test/annotations/annotation/*.txt|cut -d' ' -f1 | sort | uniq`

for i in $avi; do
  dir=`echo $i | cut -d. -f1`
  f1=`echo $dir | cut -d/ -f8`
  #  echo $dir $f1 $f2

  for j in $detection; do
      if [ $f1 = $j ]; then
          echo $dir $f1 $j
          mkdir -p /mnt/sda/diva/thumos14/data/test/frames/$f1
          rm /mnt/sda/diva/thumos14/data/test/frames/$f1/*
          ffmpeg -i $i -q:v 1 -r 25 /mnt/sda/diva/thumos14/data/test/frames/$f1/image_%5d.jpg
      fi
  done
done
