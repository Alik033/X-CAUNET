from __future__ import absolute_import, division, print_function

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import torchvision.models as pt_models
import dataset as dataset
from vgg import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from options import opt, device
from model import *
from misc import *
# from progress.bar import Bar
import re
import sys
import clip
import torchvision.transforms as T
from ssim import *
#from U_AS_transformer_v3 import U_Restormer
import wandb


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


if __name__ == '__main__':

	#scale = opt.scale

	# run_name = 'CADAqua_lol_check'
	# run = wandb.init(project="CADAqua", config=opt, name=run_name)
	transform_re = T.Resize(size = (224,224))
	transform_PIL = T.ToPILImage()
	model, preprocess = clip.load("ViT-B/32", device=device)



	print("Underwater Image Enhancement")

	netG = U_Restormer()
	netG.to(device)

	L1_loss = nn.L1Loss()
	mse_loss = nn.MSELoss()
	#ssim_loss = SSIMLoss(11)

	vgg = Vgg16(requires_grad=False).to(device)

	optim_g = optim.Adam(netG.parameters(), 
						 lr=opt.learning_rate_g, 
						 betas = (opt.beta1, opt.beta2), 
						 weight_decay=opt.wd_g)

		
	#scheduler = StepLR(optim_g, step_size=25, gamma=0.1)
	dataset = dataset.Dataset_Load(data_path = opt.data_path,
								   transform=dataset.ToTensor()
								   )
	batches = int(dataset.len / opt.batch_size)

	dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
	
	if not os.path.exists(opt.checkpoints_dir):
		os.makedirs(opt.checkpoints_dir)
	
	models_loaded = getLatestCheckpointName()    
	latest_checkpoint_G = models_loaded
	
	print('loading model for generator ', latest_checkpoint_G)
	
	if latest_checkpoint_G == None :
		start_epoch = 1
		print('No checkpoints found for netG and netD! retraining')
	
	else:
		checkpoint_g = torch.load(os.path.join(opt.checkpoints_dir, latest_checkpoint_G))    
		start_epoch = checkpoint_g['epoch'] + 1
		netG.load_state_dict(checkpoint_g['model_state_dict'])
		optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
		for param_group in optim_g.param_groups:
			param_group['lr'] = opt.learning_rate_g
			
		print('Restoring model from checkpoint ' + str(start_epoch))

	
	netG.train()
	

	for epoch in range(start_epoch, opt.end_epoch + 1):
		
		# bar = Bar('Training', max=batches)
	
		opt.total_mse_loss = 0.0
		opt.total_vgg_loss = 0.0
		opt.total_G_loss = 0.0

		for i_batch, sample_batched in enumerate(dataloader):

			hazy_batch = sample_batched['hazy']
			clean_batch = sample_batched['clean']

			hazy_batch = hazy_batch.to(device)
			clean_batch = clean_batch.to(device)

			optim_g.zero_grad()

			pred_batch = netG(hazy_batch)
			channel_wise_mul = [1, 1.5, 2]

			batch_mse_loss = torch.mul(opt.lambda_mse, L1_loss(pred_batch, clean_batch))
			batch_mse_loss.backward(retain_graph=True)

			batch_vgg_loss = 0.0
			for i in range(opt.batch_size):
				pred_image_batch = transform_re(pred_batch[i])
				pred_image_batch = transform_PIL(pred_image_batch)
				pred_image_features = preprocess(pred_image_batch).unsqueeze(0).to(device)
				clean_image_batch = transform_re(clean_batch[i])
				clean_image_batch = transform_PIL(clean_image_batch)
				clean_image_features = preprocess(clean_image_batch).unsqueeze(0).to(device)
				pred_image_features = model.encode_image(normalize_batch(pred_image_features))
				clean_image_features = model.encode_image(normalize_batch(clean_image_features))
				batch_vgg_loss += torch.mul(opt.lambda_vgg, mse_loss(pred_image_features, clean_image_features))
			batch_vgg_loss.backward(retain_graph=True)
			
			opt.batch_mse_loss = batch_mse_loss.item()
			opt.total_mse_loss += opt.batch_mse_loss


			opt.batch_vgg_loss = batch_vgg_loss.item()
			opt.total_vgg_loss += opt.batch_vgg_loss

			opt.batch_G_loss = opt.batch_mse_loss + opt.batch_vgg_loss
			opt.total_G_loss += opt.batch_G_loss
			
			optim_g.step()
			#scheduler.step()

			# bar.suffix = f' Epoch : {epoch} | ({i_batch+1}/{batches}) | ETA: {bar.eta_td} | g_mse: {opt.batch_mse_loss} | g_vgg: {opt.batch_vgg_loss}'
			print('\r Epoch : ' + str(epoch) + ' | (' + str(i_batch+1) + '/' + str(batches) + ') | mse: ' + str(opt.batch_mse_loss) + ' | clip: ' + str(opt.batch_vgg_loss), end='', flush=True)
			# bar.next()
			
		model_params = {}
		for name, param in netG.named_parameters():
			if param.requires_grad:
				model_params[name] = param.data
		

		print('\nFinished ep. %d, lr = %.6f, total_mse = %.6f, total_clip = %.6f' % (epoch, get_lr(optim_g), opt.total_mse_loss, opt.total_vgg_loss))
			# print('training epoch %d, %d / %d patches are finished, g_mse = %.6f' % (
			 # epoch, i_batch, batches, opt.batch_mse_loss))

		# wandb.log({
        #         	'epoch':epoch, 
		# 			'L1_loss':opt.total_mse_loss, 
		# 			'vgg_loss':opt.total_vgg_loss, 
		# 			'ssim_loss':opt.total_ssim_loss, 
		# 			'total_loss':opt.total_G_loss,
		# 			'alpha_r':model_params['alph'],
		# 			'beta_g':model_params['beta'],
		# 			'gamma_b':model_params['gama']
        #     })

		torch.save({'epoch':epoch, 
					'model_state_dict':netG.state_dict(), 
					'optimizer_state_dict':optim_g.state_dict(), 
					'L1_loss':opt.total_mse_loss, 
					'vgg_loss':opt.total_vgg_loss,  
					'opt':opt,
					'total_loss':opt.total_G_loss}, os.path.join(opt.checkpoints_dir, 'netG_' + str(epoch) + '.pt'))
