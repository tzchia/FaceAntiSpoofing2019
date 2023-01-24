# Face Anti-Spoofing 2019
⬇ click the preview image below to watch the demo vidoe on youtube ⬇  
[![Watch the video](https://github.com/tzchia/tzchia.github.io/blob/main/images/faceAntiSpoofing2019.jpg)](https://youtu.be/fNzQdmfk-z8)

## Data Preparation
python picFromVideo.py --input Video.mp4 --output faceDataDir --detector src/face_detector --prefix 20190520_

## Training
### version 1.0. keras + VGG
python train_keras+VGG.py --dataset faceDataDir --model models --le le.pickle --modelComplexity 43 --prep clahe --epoch 100 --side 32 --batch 256
### version 2.0. tensorflow 2.0 + ResNet
python train_tf2+ResNet.py --task 20200520 --root faceDataDir --net ResNet12
## Inferece
python inference.py --model trained.h5 --le le.pickle --detector src/face_detector --side 32 --prep clahe

## References:
1. image processing library [`imutils`](https://pypi.org/user/jrosebr1/)
2. face detector [`CAFFE-DNN`](https://github.com/gopinath-balu/computer_vision/tree/master/CAFFE_DNN)
3. anti-spoofing model arhitectures: [`VGG`](https://github.com/machrisaa/tensorflow-vgg) & [`ResNet`](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md)
