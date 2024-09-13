import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F #for activations
from torchvision import datasets, models, transforms

import time
import os
import copy
from tqdm import tqdm




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VanillaDL:
	def __init__(self, bs, data_dir, input_size, split=0.8):
		self.batchsize = bs
		self.data_dir = data_dir
		self.split = split
		self.transform =  transforms.Compose([transforms.CenterCrop(input_size),transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])])

		self.dataset = datasets.ImageFolder(self.data_dir,self.transform)
		self.train_dataset = self.dataset
		self.val_dataset = self.dataset
		
		self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batchsize,shuffle=True, num_workers=0)
		self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batchsize,shuffle=True, num_workers=0)
		self.class_names = self.dataset.classes
		lengths = [int(len(self.train_dataset)), int(len(self.val_dataset))]
		self.dataset_sizes = {'train': lengths[0], 'val': lengths[1]}



class TRAINERMAIN:
	def __init__(self):
		pass
	def train_model(self,DATA, model, criterion, optimizer, scheduler, num_epochs=25, is_inception = False):
		start = time.time()

		best_model_wts = copy.deepcopy(model.state_dict())
		best_acc = 0.0
		dataloaders ={}
		dataloaders['train'], dataloaders['val'] = DATA.train_dataloader, DATA.val_dataloader

		TR_ACCURACY=[]
		TR_LOSS=[]
		VAL_ACCURACY=[]
		VAL_LOSS=[]


		for epoch in tqdm(range(num_epochs)):
			#print('Epoch {}/{}'.format(epoch, num_epochs - 1))
			#print('-' * 10)

			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
					model.train()  # Set model to training mode
				else:
					model.eval()   # Set model to evaluate mode

				running_loss = 0.0
				running_corrects = 0

				# Iterate over data.
				for inputs, labels in dataloaders[phase]:
					inputs = inputs.to(device)
					labels = labels.to(device)
					#print(inputs.shape)
					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):
						if is_inception and phase == 'train':
							outputs, aux_outputs = model(inputs)
							loss1 = criterion(outputs, labels)
							loss2 = criterion(aux_outputs, labels)
							loss = loss1 + 0.4*loss2
						else:
							outputs = model(inputs)
							loss = criterion(outputs, labels)
						_, preds = torch.max(outputs, 1)
						# backward + optimize only if in training phase
						if phase == 'train':
							optimizer.zero_grad()
							loss.backward()
							optimizer.step()

					# statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)

				if phase == 'train':
					scheduler.step()

				epoch_loss = running_loss / DATA.dataset_sizes[phase]
				epoch_acc = running_corrects.double() / DATA.dataset_sizes[phase]

				print('{} Loss: {:.4f} Acc: {:.4f}'.format(
					phase, epoch_loss, epoch_acc))


				if phase == 'train':
					print("Training")
					TR_ACCURACY.append(epoch_acc)
					TR_LOSS.append(epoch_loss)
				else:
					print("Valuation")
					VAL_ACCURACY.append(epoch_acc)
					VAL_LOSS.append(epoch_loss)

				# deep copy the model
				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())


		time_elapsed = time.time() - start
		print('Training complete in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))

		model.load_state_dict(best_model_wts)
		return TR_ACCURACY, TR_LOSS, VAL_ACCURACY, VAL_LOSS, model


class CREATE_MODEL:
	def __init__(self, model_name, num_classes):
		self.num_classes = num_classes

		if model_name == "resnet":
			self.model = torchvision.models.resnet18(pretrained=True)
			for param in self.model.parameters():
				param.requires_grad = False
			self.num_features = self.model.fc.in_features
			self.model.fc = nn.Linear(self.num_features, self.num_classes)

		elif model_name == "inceptionv3":
			self.model = torchvision.models.inception_v3(pretrained=True)
			for param in self.model.parameters():
				param.requires_grad = False
			self.num_features = self.model.AuxLogits.fc.in_features
			self.model.AuxLogits.fc = nn.Linear(self.num_features, self.num_classes)
			self.model.fc = nn.Linear(self.model.fc.in_features,num_classes)

		elif model_name == "mobilenetv2":
			self.model = torchvision.models.mobilenet_v2(pretrained=True)
			for param in self.model.parameters():
				param.requires_grad = False
			self.num_features = self.model.classifier[1].in_features
			self.model.classifier[1] = torch.nn.Linear(self.num_features, self.num_classes)
			self.model.fc = self.model.classifier[1]

		elif model_name == "vgg":
			self.model = torchvision.models.vgg11_bn(pretrained=True)
			for param in self.model.parameters():
				param.requires_grad = False
			self.num_features = self.model.classifier[6].in_features
			self.model.classifier[6] = nn.Linear(self.num_features,self.num_classes)
			self.model.fc = self.model.classifier[6]

		elif model_name == "squeezenet":
			self.model = torchvision.models.squeezenet1_0(pretrained=True)
			for param in self.model.parameters():
				param.requires_grad = False
			self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
			self.model.num_classes = self.num_classes
			self.model.fc = self.model.classifier[1]


	def get_confusion_matrix(self, in_data, num_classes):
		confusion_matrix = torch.zeros(num_classes,num_classes)
		with torch.no_grad():
			for data in in_data:
				X_test, y_test = data[0].to(device),data[1].to(device)
				pred= self.model(X_test)
				_ , predicted = pred.max(1)
				for true_val, pred_val in zip(y_test.view(-1), predicted.view(-1)):
					confusion_matrix[true_val.long(), pred_val.long()] += 1
		self.confusion_matrix = confusion_matrix
		return self.confusion_matrix

	def test_accuracy(self, in_data):
		total = 0
		correct = 0
		with torch.no_grad():
			for data in in_data:
				X_test, y_test = data[0].to(device),data[1].to(device)
				pred= self.model(X_test)
				_ , predicted = pred.max(1)
				total += y_test.size(0)
				correct += predicted.eq(y_test).sum().item()
		return 100.*correct/total

	def evaluation_params(self, in_data):
		y_pred = []
		y_ground = []
		with torch.no_grad():
			for data in in_data:
				X_test, y_test = data[0].to(device),data[1].to(device)
				pred= self.model(X_test)
				_ , predicted = pred.max(1)
				for true_val, pred_val in zip(y_test.view(-1), predicted.view(-1)):
					y_pred.append(pred_val.item())
					y_ground.append(true_val.item())
		return y_ground, y_pred




