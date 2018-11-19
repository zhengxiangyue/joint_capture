#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from src.opt import *
from src.model import LinearModel, weight_init
from src.model2 import Model
from src.utils import update_lr, unnormalize, sort_ckpt, normalize
from torch.autograd import Variable

def initial():
	# model = LinearModel()
	model = Model()
	CUDA = torch.cuda.is_available()
	if CUDA:
		model = model.cuda()

	# cktp_file = './checkpoint/ckpt_160.pth.tar'
	cktp_file = './checkpoint/gt_ckpt_best.pth.tar'
	ckpt = torch.load(cktp_file, map_location=lambda storage, loc: storage)
	# model.load_state_dict(ckpt['model'])
	model.load_state_dict(ckpt['state_dict'])
	return CUDA, model

def pose_detection(input, model, CUDA):
	model.eval()
	data = [float(num) for num in input.split(',')]
	data = np.array(data).reshape((1, -1))

	stat_2d = torch.load('./data/stat_2d.pth.tar')
	used_data = normalize(data, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])
	used_data = np.repeat(used_data, 64, axis = 0)

	used_data = torch.from_numpy(used_data).float()

	used_data = Variable(used_data.cuda()) if CUDA else Variable(used_data.cpu())
	output = model(used_data)

	stat_3d = torch.load('./data/stat_3d.pth.tar')
	unnormal_output = unnormalize(output.data.cpu(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

	used_output = unnormal_output[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
	
	output_3d = used_output[0].reshape((1, -1))

	return output_3d
