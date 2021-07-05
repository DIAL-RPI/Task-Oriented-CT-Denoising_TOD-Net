############# Demo for task-oriented denoising network (TOD-Net) #################

This is the demo for our MICCAI-2021 paper 'Task-Oriented Low-Dose CT Image Denoising'. The original paper can be found here:
https://arxiv.org/abs/2103.13557

The codes include two parts, 'Train downstream tasks' and 'Train TOD-Net' as below.


### Part I: Train downstream tasks
Train_downstream_tasks.py -- the main file to train downstream models, unet is used as an example

dataset.py -- user defined dataloader functions (loading 3D CT images)

utils.py -- utils functions such as model initialization, model saving ect.

metric.py -- eval metrics for the downstream tasks



### Part II: Train TOD-Net
main.py --  the main file to train taks-oriented wgan network

model.py --  downstream segmentation network

TOD-Net.py -- task-oriented denoising network

loss.py -- losses used for training TOD-Net

dataset.py -- user defined dataloader functions (loading 3D CT images)

utils.py -- utils functions such as model initialization, model saving ...

metric.py -- eval metrics for the downstream tasks
