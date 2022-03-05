time=$(date "+%Y-%m-%d_%H:%M:%S")
host=$(hostname)
gpu=3

model='SPP6_3_HarmonicConv_Replace'
train_datatype='shs100k'
val_datatype='shs100k_val'

CUDA_VISIBLE_DEVICES=$gpu python -u softdtw.py neuralwarp_train --model=NeuralDTW_CNN_Mask_dilation_$model --num_workers=16 --batch_size=20 --is_label=True --is_random=True --notes='mask' --save_model=True --manner=train --params=params/neuraldtw/mask0 --test_length=400 --zo=False --train_datatype $train_datatype --val_datatype $val_datatype --lr 1e-4 --weight_decay 1e-6 --SparseL1loss 0 > test_on_ori/${host}_G${gpu}_${time}_${model}_[${train_datatype}].txt