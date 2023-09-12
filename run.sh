#!/bin/bash
python train.py \
	hello_world  \
	--data_dir ../datasets/ \
	--dataset cifar10 \
	--gpu 0 \
	--save_freq 300 \
	--model resnet18 \
	--split 1 \
	--batch_size 256 \
	--epochs 1200 \
	--encoder_dim 10 \
	--temperature 0.2 \
	--optimizer lars \
	--ours \
	--lr  0.5 \
