## Fully Convolutional Instance-aware Semantic Segmentation

This repository is copied from offcial [FCIS](https://github.com/msracver/FCIS.git).


### Installation

* Docker image can be found in docker hub which already has environment [FCIS_MXNET](https://hub.docker.com/repository/docker/smishra03/mxnet).

### Demo

1. To run the demo with our trained model (on COCO trainval35k), please download the model manually from [OneDrive](https://1drv.ms/u/s!Am-5JzdW2XHzhqMJZmVOEDgfde8_tg) (Chinese users can also get it from [BaiduYun](https://pan.baidu.com/s/1geOHioV) with code `tmd4`), and put it under folder `model/`.

	Make sure it looks like this:
	```
	./model/fcis_coco-0000.params
	```
2. Run
	```
	python ./fcis/demo.py
	```
### Training
* Follow the train.sh script for training in docker image. 
* Change the configuration file according to custom training settings such as setting all paths .