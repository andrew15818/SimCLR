python train_lincls.py \
	--data_dir ../datasets \
	--checkpoint runs/hello_world/ours_cifar10_resnet18_ratio100_split1/model.pth.tar \
	--dataset cifar10 \
	--batch_size 256 \
	--arch resnet18 \
	--split 1 \
	--ours \
