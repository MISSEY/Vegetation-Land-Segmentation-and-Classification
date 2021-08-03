cd FCIS
git clone https://github.com/msracver/FCIS.git ## clone official FCIS
cd FCIS
sh init.sh ## intialise all necessary files

## Only for testing
rm experiments/fcis/fcis_end2end_train_test.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/experiments/fcis/fcis_end2end_train_test.py experiments/fcis/

# copy the edited python files in cloned repository
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
pwd

rm fcis/config/config.py
cp /home/smishra/fcis/1/Vegetation-Land-Segmentation-and-Classification/FCIS/fcis/config/config.py fcis/config/

python experiments/fcis/fcis_end2end_train_test.py --cfg "experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml"
