import sys
import time
import os
import numpy as np
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from data import BatchData, AverageMeter
from torchvision import transforms

from replay import *
from mobilenet import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
avg_acc = 0
test_name = "EfficientNetReplay"
def clip_gradient(optimizer, grad_clip):
	"""
	Clips gradients computed during backpropagation to avoid explosion of gradients.
	:param optimizer: optimizer with the gradients to be clipped
	:param grad_clip: clip value
	"""
	for group in optimizer.param_groups:
		for param in group['params']:
			if param.grad is not None:
				param.grad.data.clamp_(-grad_clip, grad_clip)

def save_ckpt(loss, lr_log, top1, batch, net, epoch, total):
	
	torch.save(net.state_dict(), 'batch_q{}'.format(batch,epoch))
	f = open("losses.txt", "a")
	f.write('Batch{}, Epoch{}, LR:{}, Loss :{}, top1:{}'.format(batch, epoch, lr_log, loss.avg, top1.avg))
	f.write("\n")
	f.close()

def train(train_loader, val_loader, train_task, model, criterion, optimizer, epoch, par, replay, flag, args):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	val=[1]
	
	if val_loader:
		val = [data for data in replay]
	   
	print(len(val))
	for param_group in optimizer.param_groups:
		lr_log = param_group['lr']

	f = 0

	total_norm = 0
	end = time.time()
	i = -1
	if flag:
		for data in train_loader[0]:
			i+=1
			# measure data loading time
		   
			input_var, target_var = data
			input_var, target_var = input_var.to(device), target_var.to(device)
			data_time.update(time.time() - end)
			

			
			for p in par:
				total_norm += torch.sum(torch.abs(p))
			total_norm = total_norm ** (1. / 2)

			# compute output
			output = model(input_var)

			loss = criterion(output, target_var)

			# measure accuracy and record loss
			prec1= accuracy(output.data, target_var)
		   
			losses.update(loss.item())
			top1.update(prec1[0])

		   
			optimizer.zero_grad()
			loss.backward()
			
			optimizer.step()
			# print("Memory", torch.cuda.memory_allocated()/1e9, torch.cuda.max_memory_allocated()/1e9)
			

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.disp_iter == 0:
				print('Batch: {batch} '
					'Epoch: [{0}][{1}/{2}]  '
					'LR: {lr}'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
					  'Loss {loss.val:.7f} ({loss.avg:.4f})  '
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
					  'Grad:{norm}'.format(
					   epoch, i, len(train_loader[0]), batch_time=batch_time,
					   data_time=data_time, loss=losses, top1=top1, batch=train_task, norm=total_norm, lr=lr_log))

			if val_loader and len(val):
				
				input_var, target_var = val[f]
				input_var, target_var = input_var.to(device), target_var.to(device)
				output = model(input_var)
				#output = metric_fc(feature, target_var)
				loss = criterion(output, target_var) 
				# measure accuracy and record loss
				prec1= accuracy(output.data, target_var)
			   
				losses.update(loss.item())
				top1.update(prec1[0])
				
				#compute gradient and do SGD step
				f=(f +1)%(len(val))
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				if i % args.disp_iter == 0:
					print('Batch_Replay: {batch} '
						'Epoch: [{0}][{1}/{2}]  '
						'LR: {lr}'
						  'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
						  'Loss {loss.val:.7f} ({loss.avg:.4f})  '
						  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
						  'Grad:{norm}'.format(
						   epoch, i, len(train_loader[0]),
						   data_time=data_time, loss=losses, top1=top1, batch=train_task, norm=total_norm, lr=lr_log))

			   

	if len(val) > len(train_loader[0]):
		for j in range(f, len(val)):
			input_var, target_var = val[j]
			input_var, target_var = input_var.to(device), target_var.to(device)
			
			for p in par:
				total_norm += torch.sum(torch.abs(p))
			total_norm = total_norm ** (1. / 2)

			output = model(input_var)
			loss = criterion(output, target_var)
			# measure accuracy and record loss
			prec1= accuracy(output.data, target_var)
		   
			losses.update(loss.item())
			top1.update(prec1[0])
			
			optimizer.zero_grad()
			loss.backward()
			
			optimizer.step()
		   

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			if j % args.disp_iter == 0:
				print('Batch: {batch} '
					'Epoch: [{0}][{1}/{2}]  '
					'LR: {lr}'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
					  'Loss {loss.val:.7f} ({loss.avg:.4f})  '
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})  '
					  'Grad:{norm}'.format(
					   epoch, i, len(val_loader[0]), batch_time=batch_time,
					  loss=losses, top1=top1, batch=train_task, norm=total_norm, lr=lr_log))

			

	adjust_learning_rate(optimizer, epoch)
	if epoch%1 == 0:
		save_ckpt(losses, lr_log, top1, train_task, model, epoch, args.num_epoch)

	return losses


def val_acc(val_loader, model):
	acc_list = []
	top1 = AverageMeter()
	for data in val_loader[0]:
		
	
		input_var, target_var = data
		input_var, target_var = input_var.to(device), target_var.to(device)

		output = model(input_var)
	
		prec1 = accuracy(output.data, target_var)
		top1.update(prec1[0])

	return top1.avg
		

def validate(val_loader, model,  criterion, batch,task, viw, replay):
	global avg_acc
	losses = AverageMeter()
	top1 = AverageMeter()
	i = 0
	acc_list = []
   
	for data, d1 in zip(val_loader[0], viw):
		
			
		i+=1
		
		input_var, target_var = data
		input_var, target_var = input_var.to(device), target_var.to(device)

		output = model(input_var)
		loss = criterion(output, target_var)
		prec1 = accuracy(output.data, target_var)
		acc_list.append(prec1[0].cpu())
		losses.update(loss.item())
		top1.update(prec1[0])
	print("Replay:", len(replay.replaybuffer), "Task:", task)
	if batch == task:
		
		replay.update_best(list(np.array(acc_list)), task)
	   
	if batch < task:
		print(len(val_loader[1]))
		replay.update_replay(acc_list, val_loader[1], val_loader[2], batch, task)
	print('Batch:{}    '
		   'Acc:{}    '
		   'Loss:{}    '.format(batch, top1.avg, losses.avg))
	f = open("val.txt", "a")
	f.write('Trained Batch: {}'.format(task))
	f.write("\n")
	f.write('Batch{}, Loss :{}, top1:{}'.format(batch,losses.avg, top1.avg))
	f.write("\n")
	f.close()

	avg_acc = (avg_acc*(batch) + top1.avg)/(batch+1)
	print(avg_acc)
	return replay

def group_weight(module):
	group_decay = []
	group_no_decay = []
	for m in module.modules():
		if isinstance(m, nn.Linear):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, nn.modules.conv._ConvNd):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, nn.modules.batchnorm._BatchNorm):
			if m.weight is not None:
				group_no_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
	  

	assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
	param_m = group_decay + group_no_decay

	groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
	# print(groups)
	return groups, param_m




def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.98 ** (epoch))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr



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


def create_optimizer(net, args):
	grouped, par = group_weight(net)
	optimizer = torch.optim.RMSprop(grouped, lr=args.lr,weight_decay=args.weight_decay)
	return optimizer, par

def create_optimizer1(net, args):
	grouped, par = group_weight(net)
	
	optimizer = torch.optim.SGD(grouped, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1, nesterov=True)
	return optimizer, par

def main(args):

	#model = EfficientNet.from_name("efficientnet-b3").to(device)
	model = MobileNetV2().to(device)
	f = open("losses.txt", "a")
	f.write(test_name)
	f.write("\n")
	f.close()
	f = open("val.txt", "a")
	f.write(test_name)
	f.write("\n")
	f.close()
	optimizer, par = create_optimizer1(model, args)
	
	
	if args.load:
		model.load_state_dict(torch.load("batch0"))
		
	if args.focal_loss:
		loss = FocalLoss(gamma=args.gamma).to(device)
	else:
		loss = nn.CrossEntropyLoss().to(device)
	trans = transforms.Compose([
		transforms.Resize((224,224)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	])

	transview = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()])

	train_batch_list = [BatchData(args.train_dataset_path, 'train', i, trans) for i in range(1, 13)]

	train_loader_list = [(torch.utils.data.DataLoader(batch, batch_size=args.batch_size, shuffle=True, num_workers=2),  batch.data_list, batch.label_list)
						for batch in train_batch_list]

	val_batch_list = [BatchData(args.val_dataset_path, 'validation', i, trans) for i in range(1, 13)]

	val_loader_list = [(torch.utils.data.DataLoader(batch, batch_size=args.batch_size, shuffle=False, num_workers=2), batch.data_list, batch.label_list)
						for batch in val_batch_list]

	view_batch_list = [BatchData(args.val_dataset_path, 'validation', i, transview) for i in range(1, 13)]

	view_loader_list = [torch.utils.data.DataLoader(batch, batch_size=args.batch_size, shuffle=True, num_workers=2)
						for batch in view_batch_list]

	replay1 = ReplayMemory(12, len(val_loader_list[0]), 40)

	
	replay = torch.utils.data.DataLoader(ReplayData(replay1.replaybuffer, replay1.label, trans), batch_size=args.batch_size, shuffle=False, num_workers=0)
	
	for train_task in range(12):
		
		
		model.train()
		
		if train_task != -1:
			best_acc = AverageMeter()
			los = AverageMeter()
			
			if train_task == 11:
				args.num_epoch = 30

			for epoch in range(1,args.num_epoch+1):
				if train_task == 11 and epoch == 21:
					args.lr = 0.003
				if train_task ==0:
					los = train(train_loader_list[train_task], train_task>0, train_task, model, loss, optimizer, epoch, par, 0, 1, args)
				else:
					if epoch == 2:
						replay1 = validate(val_loader_list[train_task-1], model,  loss, train_task-1, train_task, view_loader_list[train_task-1], replay1)
						replay = torch.utils.data.DataLoader(ReplayData(replay1.replaybuffer, replay1.label, trans), batch_size=2*args.batch_size, shuffle=False, num_workers=0)
						print("Replay:", len(replay1.replaybuffer))

					los = train(train_loader_list[train_task], val_loader_list[train_task], train_task, model, loss, optimizer, epoch, par, replay, 1,args)
				acc = val_acc(val_loader_list[train_task], model)
				if acc > best_acc.avg:
					best_acc.update(acc)

				elif best_acc.avg - acc >= 1:
					print(">>>>>>Early Stopping:", acc, best_acc.avg)
					lr_log = 0
					for param_group in optimizer.param_groups:
						lr_log = param_group['lr']

					# save_ckpt(los, lr_log, best_acc, train_task, model, epoch, args.num_epoch)
					break
		replay1.reset()
		for i in range(train_task + 1):
			model.eval()
			with torch.no_grad():
				replay1 = validate(val_loader_list[i], model,  loss, i, train_task, view_loader_list[i], replay1)

		if train_task:

			replay = torch.utils.data.DataLoader(ReplayData(replay1.replaybuffer, replay1.label, trans), batch_size=2*args.batch_size, shuffle=True, num_workers=0)
		else:
			replay = torch.utils.data.DataLoader(ReplayData(replay1.replaybuffer, replay1.label, trans), batch_size=2*args.batch_size, shuffle=False, num_workers=0)

		



if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--train_dataset_path',
						default='./data')
	parser.add_argument('--val_dataset_path',
						default='./data')
	parser.add_argument('--load',
						default=0)
	parser.add_argument('--batch_size',
						default=16)

	parser.add_argument('--gpus', default='0,1,2',
						help='gpus to use, e.g. 0-3 or 0,1,2,3')
	parser.add_argument('--batch_size_per_gpu', default=1, type=int,
						help='input batch size')
	parser.add_argument('--num_epoch', default=20, type=int,
						help='epochs to train for')
	parser.add_argument('--start_epoch', default=1, type=int,
						help='epoch to start training. useful if continue from a checkpoint')
	parser.add_argument('--epoch_iters', default=3000, type=int,
						help='iterations of each epoch (irrelevant to batch size)')
	parser.add_argument('--optim', default='SGD', help='optimizer')
	parser.add_argument('--lr', default=0.01, type=float, help='LR')#used 0.01
	parser.add_argument('--lr_pow', default=0.9, type=float,
						help='power in poly to drop LR')
	parser.add_argument('--beta1', default=0.9, type=float,
						help='momentum for sgd, beta1 for adam')
	parser.add_argument('--weight_decay', default=0.0002, type=float,
						help='weights regularizer')#Changes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	parser.add_argument('--disp_iter', type=int, default=10,
						help='frequency to display')

	parser.add_argument('--margin-m', type=float, default=0.5, help='angular margin m')
	parser.add_argument('--margin-s', type=float, default=64.0, help='feature scale s')
	parser.add_argument('--easy-margin', type=bool, default=False, help='easy margin')
	parser.add_argument('--focal-loss', type=bool, default=False, help='focal loss')
	parser.add_argument('--gamma', type=float, default=2.0, help='focusing parameter gamma')


	args = parser.parse_args()
	print("Input arguments:")
	for key, val in vars(args).items():
		print("{:16} {}".format(key, val))

	main(args)
