#!/bin/bash
python train.py \
	--data-dir ../datasets \
	--arch resnet18 \
	--batch-size 512 \
	--dataset cifar10 \
	--num-classes 10 \
	--optimizer adam \
	--lr 3e-4 \
	--encoder_dim 512 \
	--temperature 0.2 \
	--proj_hid_dim 128 \
	--epochs 800 \
