import sys
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from mobilenet import MobileNetV2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load mobilenet without the last fc layer
net = MobileNetV2().to(device)

# determine optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# check if cuda exist then use cuda otherwise use cpu



# Construct pytorch dataset class
class BatchData(Dataset):

	def format_images(self, path, datatype, batch_index):
		path_prefix = '{}/{}/batch{}/'.format(path, datatype, batch_index)
		path_prefix = os.path.join(path,datatype,'task'+str(batch_index))+'/'
		table = pd.read_csv(path_prefix + 'labels.csv')
		table = table.sample(frac=1)
		data_list = [path + "/" + filename for filename in table['file name'].tolist()]
		label_list = table['label'].tolist()

		return data_list, label_list

	def __init__(self, path, datatype, batch_index, transforms):
		self.transforms = transforms
		self.data_list, self.label_list = self.format_images(path, datatype, batch_index)

		# print a summary
		print('Load {} batch {} have {} images '.format(datatype, batch_index, len(self.data_list)))

	def __getitem__(self, idx):
		img = self.data_list[idx]
		img = Image.open(img)
		label = int(self.label_list[idx])
		img = self.transforms(img)
		return img, label #self.data_list[idx].split('/')[-1].split('.')[0]

	def __len__(self):
		return len(self.data_list)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.initialized = False
		self.val = None
		self.avg = None
		self.sum = None
		self.count = None

	def initialize(self, val, weight):
		self.val = val
		self.avg = val
		self.sum = val * weight
		self.count = weight
		self.initialized = True

	def update(self, val, weight=1):
		if not self.initialized:
			self.initialize(val, weight)
		else:
			self.add(val, weight)

	def add(self, val, weight):
		self.val = val
		self.sum += val * weight
		self.count += weight
		self.avg = self.sum / self.count

	def value(self):
		return self.val

	def average(self):
		return self.avg



def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


avg_acc = 0

def feed(dataloader, is_training, batch):
	global avg_acc

	losses = list()
	top1 = AverageMeter()
	since = time.time()
	start = time.time()
	if is_training:
		net.train()
	else:
		num_epochs = 1
		net.eval()

	for epoch in range(num_epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		train_acc = 0.0
		i = 0
		for data in dataloader:
			i += 1
			# get the inputs

			inputs, labels= data
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()

			output = net(inputs)

			if is_training:
				optimizer.step()

			prec1 = accuracy(output.data, labels)
			top1.update(prec1[0])


		time_elapsed = time.time() - since
		since = time.time()
	avg_acc = (avg_acc*(batch) + top1.avg)/(batch+1)
	   
	print(' acc:{:.3f} '.format(
																				  top1.avg))

   
	

def test_inftime(dataloader):

  
	time_elapsed = 0	
	for data in dataloader:
	   
		inputs, labels= data
		inputs, labels = inputs.to(device), labels.to(device)
	   
		since = time.time()
		outputs = net(inputs)
		time_elapsed += time.time() - since

	return time_elapsed



def valid(validloader):
	return feed(validloader, is_training=False, is_valid=True)


def test(testloader, test_task):

	feed(testloader, is_training=False, batch=test_task)




def get_final_acc(test_loader_list):
	"""Fucntion for getting final accuracy on all the tasks and printing the mean"""
	for test_task in range(12):
		print('[Test in task{}]:'.format(test_task + 1))
		test(test_loader_list[test_task], test_task)
	print("Mean accuracy over all the tasks is:", avg_acc)

def get_infer_time(test_loader_list):
	""" Calculated avg inference time"""
	avg = 0
	print("Calculating inference time:")
	for i in range(3):
		
		t = 0
		for test_task in range(12):
			t += test_inftime(test_loader_list[test_task])
			
		avg = (avg*i + t)/(i+1)

	print("Inferene Time over all the test task is:", avg)



if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('[usage] python evaluate.py dataset_path')
		exit(0)
	dataset_path = sys.argv[1]
	
	print(dataset_path)
	# Image preprocessing
	trans = transforms.Compose([
		transforms.Resize((300,300)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])

	test_batch_list = [BatchData(dataset_path, 'test', i, trans) for i in range(1, 13)]

	test_loader_list = [torch.utils.data.DataLoader(batch, batch_size=16, shuffle=False, num_workers=2)
						for batch in test_batch_list]

# load your models

	net.load_state_dict(torch.load("batch_11_final"))
	net.eval()

	
	get_final_acc(test_loader_list)
	get_infer_time(test_loader_list)

	

