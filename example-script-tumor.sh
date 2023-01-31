#/usr/bin/bash


## training: self-supervised PaIRNet
python ./train-PaIRNet-longitudinal.py --max_epoch=3 --num_workers=1 --image_size="200,200" --image_channel=3 --targetname='timepoint' --image_dir='./data/tumor' --dataname='tumor' --selfsupervised --num_of_iters=20

## training: supervised PaIRNet
python ./train-PaIRNet-longitudinal.py --max_epoch=3 --num_workers=1 --image_size="200,200" --image_channel=3 --targetname='timepoint' --image_dir='./data/tumor' --dataname='tumor' --no-selfsupervised --num_of_iters=20

## training: baseline crosssectional regression
python ./train-baseline-crosssectional-regression.py --max_epoch=3 --num_workers=1 --image_size="200,200" --image_channel=3 --targetname='radius' --image_dir='./data/tumor' --dataname='tumor' --num_of_iters=20

### testing: weighted CAM visualization
python ./test-weightedCAM.py --image_size="200,200" --image_channel=3 --image_dir='./data/tumor' --dataname='tumor' --targetname='timepoint' --save_name='./result-model/tumor/lr0.01-b10.9-b20.999seed0/PaIRNet-self-supervised/best.pth'

## testing: correlation between predicted delta and groundtruth
python ./test-correlation.py --image_size="200,200" --image_channel=3 --image_dir='./data/tumor' --dataname='tumor' --targetname='radius' --save_name='./result-model/tumor/lr0.01-b10.9-b20.999seed0/PaIRNet-self-supervised/best.pth'


