#/usr/bin/bash


## training: self-supervised PaIRNet
python ./train-PaIRNet-longitudinal.py --max_epoch=1 --num_workers=1 --image_size="200,200" --image_channel=3 --targetname='timepoint'\
--image_dir='./data/tumor' --dataname='tumor' --selfsupervised

## training: supervised PaIRNet
python ./train-PaIRNet-longitudinal.py --max_epoch=1 --num_workers=1 --image_size="200,200" --image_channel=3 --targetname='timepoint'\
--image_dir='./data/tumor' --dataname='tumor' --no-selfsupervised

## training: baseline crosssectional regression
python ./train-baseline-crosssectional-regression.py --max_epoch=1 --num_workers=1 --image_size="200,200" --image_channel=3 --targetname='size'\
--image_dir='./data/tumor' --dataname='tumor'

### testing: weighted CAM visualization
#python ./test-weightedCAM.py --image_size="200,200" --image_channel=3 --image_dir='./data/tumor' --dataname='tumor'\
#--save_name=
#
### testing: correlation between predicted delta and groundtruth
#python ./test-correlation.py --image_size="200,200" --image_channel=3 --image_dir='./data/tumor' --dataname='tumor'\
#--save_name=


