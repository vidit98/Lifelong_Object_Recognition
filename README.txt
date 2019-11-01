Requirements

Pytorch 1.1
Python 3.5
numpy
pandas
PIL

A dockerfile is provided to make the environment

Code is tested on ubuntu 16.04 with GPU GTX1080 Ti

Download the model from the link given in the mail.
Place it in the Submition3 folder.

Create a data folder in Submition3.

There is a createdata.py file, place it in data folder

Place the train validation and test data as given below.

Submition3
	|
	|data
		|
		|createdata.py
		|train
		|validation
		|test

Place the data in train, validation, test folders with names task1, task2...task12.

run
$ cd Submition3/data
$ python createdata.py


To evaluate code use the commands given below
$ cd Submition3
$ python evaluate.py ./data

Evaluation file produce similar to given below


../data
[Test in task1]:
 acc:98.942 
[Test in task2]:
 acc:99.471 
[Test in task3]:
 acc:80.817 
[Test in task4]:
 acc:99.038 
[Test in task5]:
 acc:98.798 
[Test in task6]:
 acc:96.587 
[Test in task7]:
 acc:98.558 
[Test in task8]:
 acc:98.606 
[Test in task9]:
 acc:97.314 
[Test in task10]:
 acc:99.360 
[Test in task11]:
 acc:97.852 
[Test in task12]:
 acc:100.000 
Mean accuracy over all the tasks is: tensor(97.1120, device='cuda:0')Calculating inference time:
Inferene Time over all the test task is: 18.65184696515401  <----- Inference Time




To reproduce the results

$ cd Submition3
$ python train.py


->Calculating the inference time will take 3 to 4 minutes.
->Replay size is printed when doing validation during training.
->Model used MobileNetV2
