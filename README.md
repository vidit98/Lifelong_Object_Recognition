# Lifelong_Object_Recognition
Repo for competition track Lifelong Robotic Vision, IROS 2019.(Will update soon)
[Link to model](https://drive.google.com/open?id=14qocNtQcRAR0ayfOLp6qM3L8mXl9a1Gc)

Continual learning (CL) is the ability of a model to learn continually from a stream of data, building on what was learnt previously, hence exhibiting positive transfer, as well as being able to remember previously seen tasks. 

This task focused on new instance continual learning where number of classes across the tasks will be samw but the domain of the data will change such as occlusion, clutter etc. To know the rules in detail visit [this link](https://lifelong-robotic-vision.github.io/competition/Object-Recognition.html)

There are various methods such as dynamic architectures, regulaization based methods and replay based methods. We propose replay based method.

# Method

Given below is the summary of the proposed algorithm. For details please refer to [abstract](https://drive.google.com/file/d/18uw3fSKgSXh_Uw8jm7reRtTuoeHP--pa/view)

![Algorithm](https://drive.google.com/open?id=10qQI945cSNLUWorKf4t7N5wW-xCGjztX)

# Results

We achieved an mean accuracy of 97.01% at the end of 12th task.

![results](https://drive.google.com/open?id=1gJfC-9vaGKtG4GJZAOE_pMSLaDpazMLP)

# Future Work

This method can be combined with other methods to further reduce the replay size such as synaptic intelligence. We can also think in the direction of latent space replay or psuedo image generations using GANs.
