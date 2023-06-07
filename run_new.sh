#!/bin/bash
python train_new.py \
	test_experiment_mine \
	--data_dir ../datasets/ \
	--dataset cifar10 \
	--imb_ratio 100 \
	--split 3 \
	--gpu 0 \
	--save_freq 300 \
	--model resnet18 \
	--batch_size 256\
	--epochs 1200 \
	--encoder_dim 512 \
	--temperature 0.2 \
	--optimizer adam \
	--lr 3e-4 \
	--sdclr
