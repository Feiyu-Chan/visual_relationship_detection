import os
import collections
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def two_col_csv(filename):
	if isinstance(filename, str):
		file = pd.read_csv(filename, header=None)
	elif isinstance(filename, (list, tuple)):
		df = []
		for file in filename:
			file = pd.read_csv(file, header=None)
			df.append(file)
		file = pd.concat(df, ignore_index=True)

	file_dict = dict(zip(list(file.iloc[:, 0]), list(file.iloc[:, 1])))
	file_list = list(file.iloc[:, 0])
	return file_dict, file_list


class OpenImagesVRD(Dataset):
	def __init__(self, root_dir, transforms):
		super().__init__()
		self.root_dir = root_dir
		self.transforms = transforms
		self.cid = {'ImageID': 0,
						'LabelName1': 1,
					   'LabelName2': 2,
					   'XMin1': 3,
					   'XMax1': 4,
					   'YMin1': 5,
					   'YMax1': 6,
					   'XMin2': 7,
					   'XMax2': 8,
					   'YMin2': 9,
					   'YMax2': 10,
					   'RelationshipLabel': 11}

		cls_file = os.path.join(self.root_dir, 'challenge-2019-classes-vrd.csv')
		attr_file = os.path.join(self.root_dir, 'challenge-2019-attributes-description.csv')
		relation_file = os.path.join(self.root_dir, 'challenge-2019-relationships-description.csv')
		vrd_file = os.path.join(self.root_dir, 'challenge-2019-train-vrd.csv')

		self.cls_dict, self.cls_id = two_col_csv([cls_file, attr_file])
		self.relation_dict, self.relation_id = two_col_csv(relation_file)

		vrd_file = pd.read_csv(vrd_file, header='infer').values
		vrd_file[:, self.cid['LabelName1']] = [self.cls_id.index(l) for l in vrd_file[:, self.cid['LabelName1']]]
		vrd_file[:, self.cid['LabelName2']] = [self.cls_id.index(l) for l in vrd_file[:, self.cid['LabelName2']]]
		vrd_file[:, self.cid['RelationshipLabel']] = [self.relation_id.index(r)
														 for r in vrd_file[:, self.cid['RelationshipLabel']]]
		# print(vrd_file, type(vrd_file), vrd_file.shape, len(vrd_file))
		self.data = vrd_file

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_fp = os.path.join(self.root_dir, 'train', self.data[idx, self.cid['ImageID']] + '.jpg')
		try:
			img = Image.open(img_fp).convert('RGB')
		except:
			img = Image.new('RGB', (60, 30), color='black')
		if self.transforms is None:
			transformations = transforms.Compose[
				transforms.ToTensor(),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			]
			img = transformations(img)
		else:
			img = self.transforms(img)

		# return {'image': img,
		#         'label1': self.data[idx, self.cid['LabelName1']],
		#         'label2': self.data[idx, self.cid['LabelName2']],
		#         'relationship': self.data[idx, self.cid['RelationshipLabel']],
		# 		'xmin1': self.data[idx, self.cid['XMin1']],
		# 		'xmax1': self.data[idx, self.cid['XMax1']],
		# 		'ymin1': self.data[idx, self.cid['YMin1']],
		# 		'ymax1': self.data[idx, self.cid['YMax1']],
		# 		'xmin2': self.data[idx, self.cid['XMin2']],
		# 		'xmax2': self.data[idx, self.cid['XMax2']],
		# 		'ymin2': self.data[idx, self.cid['YMin2']],
		# 		'ymax2': self.data[idx, self.cid['YMax2']],
		#         }

		# return img, {'labels': torch.Tensor((int(self.data[idx, self.cid['LabelName1']]),
		# 								   int(self.data[idx, self.cid['LabelName2']]))).long(),
		# 		'image_id': torch.IntTensor((idx)),
		# 		'boxes': torch.FloatTensor(((self.data[idx, self.cid['XMin1']], self.data[idx, self.cid['YMin1']],
		# 									 self.data[idx, self.cid['XMax1']], self.data[idx, self.cid['YMax1']]),
		# 									(self.data[idx, self.cid['XMin2']], self.data[idx, self.cid['YMin2']],
		# 									 self.data[idx, self.cid['XMax2']], self.data[idx, self.cid['YMax2']]))),
		# 		# 'relationship': self.data[idx, self.cid['RelationshipLabel']],
		# 		'is_crowd': torch.zeros((2,), dtype=torch.int64),
		# 		'area': torch.ones((2,), dtype=torch.float32),
		# 		'masks': img
		# 		}
		sample = {}
		c, h, w = img.shape
		sample['img'] = img
		sample['boxes'] = torch.FloatTensor(((self.data[idx, self.cid['XMin1']] * w,
											  self.data[idx, self.cid['YMin1']] * h,
											  self.data[idx, self.cid['XMax1']] * w,
											  self.data[idx, self.cid['YMax1']] * h),
											 (self.data[idx, self.cid['XMin2']] * w,
											  self.data[idx, self.cid['YMin2']] * h,
											  self.data[idx, self.cid['XMax2']] * w,
											  self.data[idx, self.cid['YMax2']] * h)))

		sample['labels'] = torch.Tensor((int(self.data[idx, self.cid['LabelName1']]),
										 int(self.data[idx, self.cid['LabelName2']]))).long()
		sample['relationship'] = self.data[idx, self.cid['RelationshipLabel']]

		return sample



def get_dataloader(root_dir, transforms, batch_size, shuffle, num_workers, pin_memory, drop_last, timeout):
	dataset = OpenImagesVRD(root_dir=root_dir, transforms=transforms)
	dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
							pin_memory=pin_memory, drop_last=drop_last, timeout=timeout)
	return dataloader