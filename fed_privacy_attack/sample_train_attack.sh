#!/bin/bash


rm -rf evals_sites/*
rm -rf evals_sites_npy/*
rm -rf checkpoints_attackers/*



i=0
clients=21
local_epochs=10
client_frac=0.99


site_idx=0
v=0

use_iid_data=False
fp16=True
batch_size=128


fraction_of_train_used=1.0 # (public data/ points to only 10% of the dataset, so this corresponds to 10% exposed)

#model_subdir="fl_noniid/otter/epoch200global.h5"

#Base FL trained model
model_subdir="cds/nonoise_clip5/3D_dpsgd_cds_l2normclip_5.0_noisemult_0.0001_iiddata_True_nummicrobatches_1_fp16_True.h5"
model_id=12
gpu_id=0
CUDA_VISIBLE_DEVICES=$gpu_id python train_attacker_predict_sites.py 0 $batch_size $fp16 10 0.9 False $i 0 $clients $local_epochs \
 $use_iid_data $model_id $model_subdir $fraction_of_train_used &


# DP-SGD trained FL model
model_subdir="fl_dpsgd/k0_FL_noniid_dpsgd_noise0.1.h5"
model_id=14
gpu_id=1
CUDA_VISIBLE_DEVICES=$gpu_id python train_attacker_predict_sites.py 0 $batch_size $fp16 10 0.9 False $i 0 $clients $local_epochs \
 $use_iid_data $model_id $model_subdir $fraction_of_train_used &


# Randomly init model
model_subdir="random/random.h5"
model_id=16
gpu_id=3
CUDA_VISIBLE_DEVICES=$gpu_id python train_attacker_predict_sites.py 0 $batch_size $fp16 10 0.9 False $i 0 $clients $local_epochs \
 $use_iid_data $model_id $model_subdir $fraction_of_train_used &


