from object_dataloader import OpenImagesVRD
from torchvision import transforms
import torch
import torchvision
from helper import utils
from model import BaseModel, LSTMModel, TransformerModel, PriorLSTMModel
from tqdm import tqdm
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
train_fraction = 0.35
test_fraction_from_train = 0.2
lr = 3e-4
momentum = 0.9
weight_decay = 1e-4
epochs = 10
eval_freq = 1


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# use our dataset and defined transformations
dataset_train = OpenImagesVRD(root_dir=root_data_dir, transforms=transformations, train_fraction=train_fraction)
dataset_test = OpenImagesVRD(root_dir=root_data_dir, transforms=transformations, train_fraction=train_fraction)

dataset_len = len(dataset_train)
class_list = dataset_train.cls_id
num_objects = len(class_list)
relation_list = dataset_train.relation_id
num_relations = len(relation_list)

# split the dataset in train and test set
indices = torch.randperm(dataset_len).tolist()
split_index = int(test_fraction_from_train * dataset_len)
dataset = torch.utils.data.Subset(dataset_train, indices[:-split_index])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-split_index:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
	dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory,
	drop_last=drop_last, timeout=timeout)

data_loader_test = torch.utils.data.DataLoader(
	dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
	drop_last=drop_last, timeout=timeout)

#
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to('cuda')
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
# 	in_features, num_classes).to('cuda')

# model = AddVrdMlp(model, num_relations).to('cuda')

# model = BaseModel(num_objects=num_objects, num_relations=num_relations).to(device)
# model = LSTMModel(num_objects=num_objects, num_relations=num_relations).to(device)
# model = TransformerModel(num_objects=num_objects, num_relations=num_relations).to(device)
model = PriorLSTMModel(num_objects=num_objects, num_relations=num_relations).to(device)
print(model)
model.train()




params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(
# 	params, lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
optimizer.zero_grad()

def evaluate(model, data_loader_test, dataset, epoch):
	model.eval()
	dataset_from_train = dataset
	cls_dict, cls_id = dataset_from_train.cls_dict, dataset_from_train.cls_id
	print(cls_dict, cls_id)
	total = 0
	correct = {'obj1': 0, 'obj2': 0, 'rel': 0}
	for i, batch in enumerate(data_loader_test):
		# img = batch['img'].to(device)
		labels = batch['labels'].to(device)
		# boxes = batch['boxes'].to(device)
		relationship = batch['relationship'].long().to(device)
		label1 = batch['label1'].long().to(device)
		label2 = batch['label2'].long().to(device)
		obj1 = batch['obj1'].to(device)
		obj2 = batch['obj2'].to(device)
		rel = batch['rel'].to(device)
		distance = batch['distance'].to(device) # Used by only prior model
		orientation = batch['orientation'].to(device) # Used by only prior model
		predictions = model(images={'obj1': obj1, 'obj2': obj2, 'rel': rel, 'distance': distance,
									'orientation': orientation},
							targets={'obj1': label1, 'obj2': label2, 'rel': relationship})

		l_rel = predictions['conf_rel']
		l_obj1 = predictions['conf_obj1']
		l_obj2 = predictions['conf_obj2']

		_, score_obj1 = torch.max(l_obj1, 1)
		score_obj1 = (score_obj1 == label1).sum().item()
		_, score_obj2 = torch.max(l_obj2, 1)
		score_obj2 = (score_obj2 == label2).sum().item()
		_, score_rel = torch.max(l_rel, 1)
		score_rel = (score_rel == relationship).sum().item()

		total += l_rel.size(0)
		correct['obj1'] += score_obj1
		correct['obj2'] += score_obj2
		correct['rel'] += score_rel

		# print('TEST: TrEpoch, Iter, ImageID, Predictions', epoch, i, batch['name'], predictions)

		# # vis_img = img[0].cpu().numpy()
		# # print(vis_img.shape)
		# plt.imsave('results/pred_{}.png'.format(i), vis_img.transpose((1, 2, 0)))
	print('TEST: RESULTS obj1, obj2, rel:', correct['obj1']/total, correct['obj2']/total, correct['rel']/total)
	model.train()


def main():
	print('dl len', len(data_loader))
	for epoch in range(epochs):
		for i, batch in tqdm(enumerate(data_loader)):
			# img = batch['img'].to(device)
			labels = batch['labels'].to(device)
			# boxes = batch['boxes'].to(device)
			relationship = batch['relationship'].long().to(device)
			label1 = batch['label1'].long().to(device)
			label2 = batch['label2'].long().to(device)
			obj1 = batch['obj1'].to(device)
			obj2 = batch['obj2'].to(device)
			rel = batch['rel'].to(device)
			distance = batch['distance'].to(device)  # Used by only prior model
			orientation = batch['orientation'].to(device)  # Used by only prior model
			loss = model(images={'obj1': obj1, 'obj2': obj2, 'rel': rel, 'distance': distance,
									'orientation': orientation},
								targets={'obj1': label1, 'obj2': label2, 'rel': relationship})

			# list_of_dicts = []
			# for i in range(boxes.size(0)):
			# 	list_of_dicts.append({'labels': labels[i], 'boxes': boxes[i]})
			#
			# # loss_dict = model(img, list_of_dicts, relationship)
			# loss_dict = model(img, list_of_dicts)
			# loss = 0.0
			# for k, v in loss_dict.items():
			# 	loss += v
			# print(loss_dict, loss.item())

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			# print("TRAIN: Epoch, iter, loss", epoch, i, loss.item())

		torch.save({
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict()
		}, 'saved/prior_lstm_{}.pth.tar'.format(epoch))

		if (epoch+1) % eval_freq == 0:
			evaluate(model, data_loader_test, dataset_train, epoch)


if __name__ == "__main__":
	main()
