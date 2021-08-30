
#This is file is configured for training on DFKI cluster, please change the initials in the absolute path according to your
#system.


cd FCIS
## Run the file for training the FCIS, before training, one must change below files in this repo according to training:
# 1. FCIS/experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
# 2. FCIS/fcis/config/config.py

# Clone the official repo of FCIS
git clone https://github.com/msracver/FCIS.git ## clone official FCIS
cd FCIS
sh init.sh ## intialise all necessary files

# copy the customised python files in official cloned repository

# Uncomment the below line if you changed the fcis_end2end_train_test.py to only test the model on docker
#rm experiments/fcis/fcis_end2end_train_test.py
#cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/experiments/fcis/fcis_end2end_train_test.py experiments/fcis/


rm lib/dataset/coco.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/lib/dataset/coco.py lib/dataset/

rm lib/utils/load_data.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/lib/utils/load_data.py lib/utils/

rm fcis/core/module.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/fcis/core/module.py fcis/core/

rm fcis/function/test_fcis.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/fcis/function/test_fcis.py fcis/function/

rm fcis/train_end2end.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/fcis/train_end2end.py fcis/

rm experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml experiments/fcis/cfgs/

cat experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml

rm fcis/config/config.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/fcis/config/config.py fcis/config/


## Run the python script for training and testing using configuration yaml file
#python experiments/fcis/fcis_end2end_train_test.py --cfg "experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml"



## For demo
rm -rf demo
cp -R /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/demo .

rm experiments/fcis/cfgs/fcis_coco_demo.yaml
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/experiments/fcis/cfgs/fcis_coco_demo.yaml experiments/fcis/cfgs/

rm fcis/demo.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/fcis/demo.py fcis/

rm lib/utils/show_masks.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/lib/utils/show_masks.py lib/utils/

python ./fcis/demo.py