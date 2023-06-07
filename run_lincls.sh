python train_lincls_new.py \
	--data_dir ../datasets \
	--checkpoint runs/itest_experiment_mine/sdclr_cifar10_resnet18_ratio100_split2/model.pth.tar \
	--dataset cifar10 \
	--batch_size 256 \
	--arch resnet18 \
	--split 2 \
	--sdclr \
