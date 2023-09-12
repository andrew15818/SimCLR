#!/bin/bash
python train.py simclr_inat\
	--data_dir ../../datasets \
	--model resnet50 \
	--epochs 90 \
	--lr 5.0 \
	--batch-size 64 \
	--encoder_dim 8142 \
	--checkpoint ../runs/simclr_inat/lotar_inat18/model.pth \
	--momentum 0.9 \
	--gpu 1 \
	--ours \
