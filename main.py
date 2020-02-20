from dataloader import OpenImagesVRD, get_dataloader
from torchvision import transforms
import torch
import torchvision
from helper import utils
from model import AddVrdMlp
import matplotlib.pyplot as plt

root_data_dir='/home/kartik/Desktop/visual_relationship/data/open_images'
root_data_dir='/home/kjain/ml_3/data/open_images'
im_size = 256
transformations = transforms.Compose([
	transforms.Resize(im_size),
	transforms.CenterCrop(im_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Skipping since model expects [0,1]
])
batch_size = 32
shuffle = True
num_workers = 8
pin_memory = True
drop_last = True
timeout = 0 # In seconds I think
lr = 3e-4
momentum = 0.9
weight_decay = 1e-4
train_fraction = 0.2
epochs = 10
eval_freq = 1


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 62 + 1
# use our dataset and defined transformations
dataset_train = OpenImagesVRD(root_dir=root_data_dir, transforms=transformations, train_fraction=train_fraction)
dataset_test = OpenImagesVRD(root_dir=root_data_dir, transforms=transformations, train_fraction=train_fraction)
# split the dataset in train and test set
indices = torch.randperm(len(dataset_train)).tolist()
dataset = torch.utils.data.Subset(dataset_train, indices[:-2000])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-2000:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
	dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
	drop_last=drop_last, timeout=timeout)

data_loader_test = torch.utils.data.DataLoader(
	dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
	drop_last=drop_last, timeout=timeout)


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to('cuda')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
	in_features, num_classes).to('cuda')

num_relations = len(dataset_train.relation_dict)
# model = AddVrdMlp(model, num_relations).to('cuda')



print(model)
model.train()




params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
# 	params, lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

def evaluate(model, data_loader_test, dataset):
	model.eval()
	dataset_from_train = dataset
	cls_dict, cls_id = dataset_from_train.cls_dict, dataset_from_train.cls_id
	print(cls_dict, cls_id)
	for i, batch in enumerate(data_loader_test):
		img = batch['img'].to(device)
		labels = batch['labels'].to(device)
		boxes = batch['boxes'].to(device)
		relationship = batch['relationship'].long().to(device)
		predictions = model(img)
		print(predictions)
		vis_img = img[0].cpu().numpy()
		print(vis_img.shape)
		plt.imsave('results/pred_{}.png'.format(i), vis_img.transpose((1, 2, 0)))

	model.train()


def main():
	print('dl len', len(data_loader))
	for epoch in range(epochs):
		for i, batch in enumerate(data_loader):
			# print(batch)
			img = batch['img'].to(device)
			labels = batch['labels'].to(device)
			boxes = batch['boxes'].to(device)
			relationship = batch['relationship'].long().to(device)

			list_of_dicts = []
			for i in range(boxes.size(0)):
				list_of_dicts.append({'labels': labels[i], 'boxes': boxes[i]})

			# loss_dict = model(img, list_of_dicts, relationship)
			loss_dict = model(img, list_of_dicts)
			loss = 0.0
			for k, v in loss_dict.items():
				loss += v
			print(loss_dict, loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	if (epoch+1) % eval_freq == 0:
		torch.save({
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict()
		}, 'saved/detection_faster_rcnn_{}.pth.tar'.format(epoch))
		evaluate(model, data_loader_test, dataset_train)









#
# import os
# import numpy as np
# import torch
# from PIL import Image
#
# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
#
# from helper.engine import train_one_epoch, evaluate
# import helper.utils
# import helper.transforms as T
#
# def get_model_instance_segmentation(num_classes):
# 	# load an instance segmentation model pre-trained pre-trained on COCO
# 	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#
# 	# get number of input features for the classifier
# 	in_features = model.roi_heads.box_predictor.cls_score.in_features
# 	# replace the pre-trained head with a new one
# 	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
# 	# now get the number of input features for the mask classifier
# 	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# 	hidden_layer = 256
# 	# and replace the mask predictor with a new one
# 	model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
# 													   hidden_layer,
# 													   num_classes)
#
# 	return model
#
#
# def get_transform(train):
# 	transforms = []
# 	transforms.append(T.ToTensor())
# 	if train:
# 		transforms.append(T.RandomHorizontalFlip(0.5))
# 	return T.Compose(transforms)
#
#
# def main():
# 	# train on the GPU or on the CPU, if a GPU is not available
# 	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
# 	# our dataset has two classes only - background and person
# 	num_classes = 63
# 	# use our dataset and defined transformations
# 	dataset = OpenImagesVRD(root_dir=root_data_dir, transforms=transformations)
# 	dataset_test = OpenImagesVRD(root_dir=root_data_dir, transforms=transformations)
# 	# split the dataset in train and test set
# 	indices = torch.randperm(len(dataset)).tolist()
# 	dataset = torch.utils.data.Subset(dataset, indices[:-2000])
# 	dataset_test = torch.utils.data.Subset(dataset_test, indices[-2000:])
#
# 	# define training and validation data loaders
# 	data_loader = torch.utils.data.DataLoader(
# 		dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
# 		drop_last=drop_last, timeout=timeout, collate_fn=helper.utils.collate_fn)
#
# 	data_loader_test = torch.utils.data.DataLoader(
# 		dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
# 		drop_last=drop_last, timeout=timeout, collate_fn=helper.utils.collate_fn)
#
#
# 	# get the model using our helper function
# 	model = get_model_instance_segmentation(num_classes)
#
# 	# move model to the right device
# 	model.to(device)
#
# 	# construct an optimizer
# 	params = [p for p in model.parameters() if p.requires_grad]
# 	optimizer = torch.optim.SGD(params, lr=0.005,
# 								momentum=0.9, weight_decay=0.0005)
# 	# and a learning rate scheduler
# 	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
# 												   step_size=3,
# 												   gamma=0.1)
#
# 	# let's train it for 10 epochs
# 	num_epochs = 10
#
# 	for epoch in range(num_epochs):
# 		# train for one epoch, printing every 10 iterations
# 		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
# 		# update the learning rate
# 		lr_scheduler.step()
# 		# evaluate on the test dataset
# 		evaluate(model, data_loader_test, device=device)
#
# 	print("That's it!")


if __name__ == "__main__":
	main()