#!/bin/bash
python train.py \
	--arch resnet18 \
	--batch-size 256 \
	--dataset cifar10 \
	--lr 0.0003 \
	--encoder_dim 512 \
	--proj_hid_dim 128 \
	--num-classes 10 \
	--epochs 100 \
	--temperature 0.5
