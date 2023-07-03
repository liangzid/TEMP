#!/bin/bash

echo "beginning to run normal with 0.8"
echo "----------1----------"
bash 0.0.allPipeline.sh >1124-1.log

echo "beginning to run 30 epochs with 0.8"
echo "----------2----------"
bash 0.1.trainTemp.sh >1124-2.log

echo "beginning to run forward sequence"
echo "----------3----------"
bash 1.0.temperingLearning.sh >1124-3.log

echo "beginning to run backward sequence"
echo "----------4----------"
bash 1.1.temperingTraining.sh >1124-4.log

echo ">>>ALL THINGS DONE.<<<"
