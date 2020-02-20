import torch
import torchvision

class AddVrdMlp(torch.nn.Module):
	def __init__(self, original_model, num_relations):
		super().__init__()
		self.features = torch.nn.Sequential(*list(original_model.children())[:-2])
		print('features', self.features)
		self.rest = torch.nn.Sequential(*list(original_model.children())[-2:])
		print('rest', self.rest)
		self.vrd = torch.nn.Sequential(
			torch.nn.Linear(1024, 512),
			torch.nn.ReLU(),
			torch.nn.BatchNorm1d(512),
			torch.nn.Linear(512, num_relations)
		)
		print('vrd', self.vrd)

	def forward(self, x, targets, target_relationship):
		features = self.features(x, targets)
		output_loss_dict = self.rest(features)
		vrd_output = self.vrd(features)

		rel_loss = torch.nn.CrossEntropyLoss(vrd_output, target_relationship)
		output_loss_dict['relationship'] = rel_loss

		return output_loss_dict


class BaseModel(torch.nn.Module):
	def __init__(self, num_objects, num_relations):
		super().__init__()
		self.object_features = torchvision.models.resnet50(pretrained=True)
		self.object_features = torch.nn.Sequential(*list(self.object_features.children())[:-1])
		self.relationship_features = torchvision.models.resnet50(pretrained=True)
		self.relationship_features = torch.nn.Sequential(*list(self.relationship_features.children())[:-1])

		self.object_classifier = torch.nn.Linear(in_features=2048, out_features=num_objects)
		# self.relationship_classifier = torch.nn.Linear(in_features=2048*3, out_features=num_relations)
		self.relationship_classifier = torch.nn.Linear(in_features=2048, out_features=num_relations)

	def forward(self, images, targets):
		batch_size = images['rel'].size(0)
		f_obj1 = self.object_features(images['obj1'])
		f_obj1 = f_obj1.reshape((batch_size, -1))
		f_obj2 = self.object_features(images['obj2'])
		f_obj2 = f_obj2.reshape((batch_size, -1))
		f_rel = self.relationship_features(images['rel'])
		f_rel = f_rel.reshape((batch_size, -1))

		l_obj1 = self.object_classifier(f_obj1)
		l_obj2 = self.object_classifier(f_obj2)
		# l_rel = self.relationship_classifier(torch.cat((f_obj1, f_rel, f_obj2), dim=1))
		l_rel = self.relationship_classifier(f_rel)

		if self.training == True:
			loss = 0.0
			loss += self.cross_entropy_loss(l_obj1, targets['obj1'])
			loss += self.cross_entropy_loss(l_obj2, targets['obj2'])
			loss += self.cross_entropy_loss(l_rel, targets['rel'])
			return loss
		else: # Eval mode
			l_obj1 = torch.nn.Softmax(dim=1)(l_obj1)
			l_obj2 = torch.nn.Softmax(dim=1)(l_obj2)
			l_rel = torch.nn.Softmax(dim=1)(l_rel)
			return {'conf_obj1': l_obj1, 'conf_obj2': l_obj2, 'conf_rel': l_rel}

	def cross_entropy_loss(self, predictions, targets):
		return torch.nn.CrossEntropyLoss()(predictions, targets)


class LSTMModel(torch.nn.Module):
	def __init__(self, num_objects, num_relations):
		super().__init__()
		self.object_features = torchvision.models.resnet50(pretrained=True)
		self.object_features = torch.nn.Sequential(*list(self.object_features.children())[:-1])
		self.relationship_features = torchvision.models.resnet50(pretrained=True)
		self.relationship_features = torch.nn.Sequential(*list(self.relationship_features.children())[:-1])

		self.object_classifier = torch.nn.Linear(in_features=2048, out_features=num_objects)
		self.relationship_classifier = torch.nn.Linear(in_features=2048, out_features=num_relations)
		self.lstm = torch.nn.LSTM(input_size=2048, hidden_size=2048, num_layers=3, batch_first=False)

	def forward(self, images, targets):
		batch_size = images['rel'].size(0)
		f_obj1 = self.object_features(images['obj1'])
		f_obj1 = f_obj1.reshape((batch_size, -1))
		f_obj2 = self.object_features(images['obj2'])
		f_obj2 = f_obj2.reshape((batch_size, -1))
		f_rel = self.relationship_features(images['rel'])
		f_rel = f_rel.reshape((batch_size, -1))

		sequence = torch.stack([f_obj1, f_rel, f_obj2])
		outputs, _ = self.lstm(sequence) # Outputs is sequence * batch * dims

		l_obj1 = self.object_classifier(outputs[0])
		l_obj2 = self.object_classifier(outputs[2])
		l_rel = self.relationship_classifier(outputs[1])

		if self.training == True:
			loss = 0.0
			loss += self.cross_entropy_loss(l_obj1, targets['obj1'])
			loss += self.cross_entropy_loss(l_obj2, targets['obj2'])
			loss += self.cross_entropy_loss(l_rel, targets['rel'])
			return loss
		else: # Eval mode
			l_obj1 = torch.nn.Softmax(dim=1)(l_obj1)
			l_obj2 = torch.nn.Softmax(dim=1)(l_obj2)
			l_rel = torch.nn.Softmax(dim=1)(l_rel)
			return {'conf_obj1': l_obj1, 'conf_obj2': l_obj2, 'conf_rel': l_rel}

	def cross_entropy_loss(self, predictions, targets):
		return torch.nn.CrossEntropyLoss()(predictions, targets)


class TransformerModel(torch.nn.Module):
	def __init__(self, num_objects, num_relations):
		super().__init__()
		self.object_features = torchvision.models.resnet50(pretrained=True)
		self.object_features = torch.nn.Sequential(*list(self.object_features.children())[:-1])
		self.relationship_features = torchvision.models.resnet50(pretrained=True)
		self.relationship_features = torch.nn.Sequential(*list(self.relationship_features.children())[:-1])

		self.object_classifier = torch.nn.Linear(in_features=2048, out_features=num_objects)
		self.relationship_classifier = torch.nn.Linear(in_features=2048, out_features=num_relations)
		self.transformer = torch.nn.Transformer(d_model=2048, nhead=4, num_encoder_layers=1, num_decoder_layers=1,
												dim_feedforward=2048)

	def forward(self, images, targets):
		batch_size = images['rel'].size(0)
		f_obj1 = self.object_features(images['obj1'])
		f_obj1 = f_obj1.reshape((batch_size, -1))
		f_obj2 = self.object_features(images['obj2'])
		f_obj2 = f_obj2.reshape((batch_size, -1))
		f_rel = self.relationship_features(images['rel'])
		f_rel = f_rel.reshape((batch_size, -1))

		sequence = torch.stack([f_obj1, f_rel, f_obj2])
		outputs = self.transformer(sequence, sequence) # Outputs is sequence * batch * dims

		l_obj1 = self.object_classifier(outputs[0])
		l_obj2 = self.object_classifier(outputs[2])
		l_rel = self.relationship_classifier(outputs[1])

		if self.training == True:
			loss = 0.0
			loss += self.cross_entropy_loss(l_obj1, targets['obj1'])
			loss += self.cross_entropy_loss(l_obj2, targets['obj2'])
			loss += self.cross_entropy_loss(l_rel, targets['rel'])
			return loss
		else: # Eval mode
			l_obj1 = torch.nn.Softmax(dim=1)(l_obj1)
			l_obj2 = torch.nn.Softmax(dim=1)(l_obj2)
			l_rel = torch.nn.Softmax(dim=1)(l_rel)
			return {'conf_obj1': l_obj1, 'conf_obj2': l_obj2, 'conf_rel': l_rel}

	def cross_entropy_loss(self, predictions, targets):
		return torch.nn.CrossEntropyLoss()(predictions, targets)

class PriorLSTMModel(torch.nn.Module):
	def __init__(self, num_objects, num_relations):
		super().__init__()
		self.object_features = torchvision.models.resnet50(pretrained=True)
		self.object_features = torch.nn.Sequential(*list(self.object_features.children())[:-1])
		self.relationship_features = torchvision.models.resnet50(pretrained=True)
		self.relationship_features = torch.nn.Sequential(*list(self.relationship_features.children())[:-1])

		self.object_classifier = torch.nn.Linear(in_features=2048, out_features=num_objects)
		self.relationship_classifier = torch.nn.Linear(in_features=2048+2, out_features=num_relations) # For prior
		self.lstm = torch.nn.LSTM(input_size=2048, hidden_size=2048, num_layers=3, batch_first=False)

	def forward(self, images, targets):
		batch_size = images['rel'].size(0)
		f_obj1 = self.object_features(images['obj1'])
		f_obj1 = f_obj1.reshape((batch_size, -1))
		f_obj2 = self.object_features(images['obj2'])
		f_obj2 = f_obj2.reshape((batch_size, -1))
		f_rel = self.relationship_features(images['rel'])
		f_rel = f_rel.reshape((batch_size, -1))

		sequence = torch.stack([f_obj1, f_rel, f_obj2])
		outputs, _ = self.lstm(sequence) # Outputs is sequence * batch * dims

		l_obj1 = self.object_classifier(outputs[0])
		l_obj2 = self.object_classifier(outputs[2])
		l_rel = self.relationship_classifier(torch.cat([outputs[1], images['distance'].unsqueeze(dim=1),
														images['orientation'].unsqueeze(dim=1)], dim=1))

		if self.training == True:
			loss = 0.0
			loss += self.cross_entropy_loss(l_obj1, targets['obj1'])
			loss += self.cross_entropy_loss(l_obj2, targets['obj2'])
			loss += self.cross_entropy_loss(l_rel, targets['rel'])
			return loss
		else: # Eval mode
			l_obj1 = torch.nn.Softmax(dim=1)(l_obj1)
			l_obj2 = torch.nn.Softmax(dim=1)(l_obj2)
			l_rel = torch.nn.Softmax(dim=1)(l_rel)
			return {'conf_obj1': l_obj1, 'conf_obj2': l_obj2, 'conf_rel': l_rel}

	def cross_entropy_loss(self, predictions, targets):
		return torch.nn.CrossEntropyLoss()(predictions, targets)