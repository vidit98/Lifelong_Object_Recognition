import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ReplayMemory(object):

	def __init__(self, noTask, maxSize, topk):
		
		self.best = list(np.zeros((noTask, maxSize)))
		self.replaybuffer = []
		self.label = []
		self.topk = topk
		
	def reset(self):
		self.replaybuffer = []
		self.label = []
		
		
	def update_best(self,current,  task):
		
		self.best[task] = current
		#print("Print inside", self.best[task])
	def update_replay(self, current, task_list, label_list, task, trained_task):
		
		interfere = np.array(current) - np.array(self.best[task])
		#print("Interfere:", interfere)
		idx = np.argsort(interfere)
		
		#f = open("replay" + str(trained_task) +".txt", "a")
		for i in range(self.topk):
			for j in range(16*int(idx[-1-i]),16*int(idx[-1-i]) + 16):
				if j >= len(task_list):
					continue
				if task_list[j] not in self.replaybuffer:
					
					self.replaybuffer += [task_list[j]]
					self.label += [label_list[j]]
					#f.write(task_list[j] + "," + str(label_list[j])+"\n")
	
			
		#f.close()


		
class ReplayData(Dataset):
	def __init__(self, data_list, label_list, trans):
		self.trans = trans
		self.data_list = data_list
		self.label_list = label_list
	def __getitem__(self, idx):
		
		img = self.data_list[idx]
		img = Image.open(img)
		label = int(self.label_list[idx])
		img = self.trans(img)
		return img, label 
	def __len__(self):
		return len(self.data_list)


	



