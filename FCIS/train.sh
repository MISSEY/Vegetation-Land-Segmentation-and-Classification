git clone https://github.com/msracver/FCIS.git ## clone official FCIS
cd FCIS
sh init.sh ## intialise all necessary files

# copy the edited python files in cloned repository
rm lib/dataset/coco.py
cp /home/smishra/fcis/1/FCIS_mxnet/lib/dataset/coco.py lib/dataset/

rm lib/utils/load_data.py
cp /home/smishra/fcis/1/FCIS_mxnet/lib/utils/load_data.py lib/utils/

rm fcis/core/module.py
cp /home/smishra/fcis/1/FCIS_mxnet/fcis/core/module.py fcis/core/

rm fcis/function/test_fcis.py
cp /home/smishra/fcis/1/FCIS_mxnet/fcis/function/test_fcis.py fcis/function/

rm fcis/train_end2end.py
cp /home/smishra/fcis/1/FCIS_mxnet/fcis/train_end2end.py fcis/

rm experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
cp /home/smishra/fcis/1/FCIS_mxnet/experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml experiments/fcis/cfgs/

cat experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
pwd

rm fcis/config/config.py
cp /home/smishra/fcis/1/FCIS_mxnet/fcis/config/config.py fcis/config/

python experiments/fcis/fcis_end2end_train_test.py --cfg "experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml"
