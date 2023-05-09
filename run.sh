#!/bin/bash
python train.py \
	--data-dir ../datasets \
	--arch resnet18 \
	--batch-size 512 \
	--dataset cifar100 \
	--num-classes 100 \
	--optimizer adam \
	--lr .0003 \
	--encoder_dim 512 \
	--temperature 0.2 \
	--proj_hid_dim 128 \
	--resume runs/cifar100_1200_normreweight.pth.tar \
	--epochs 1500 \
