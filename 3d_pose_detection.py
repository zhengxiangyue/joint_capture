#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from src.dataset import Human36m
from src.opt import *
from src.model import Model, weight_init
from src.draw_picture import draw_picture
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn


def sort_ckpt(ckpt_dir):
	file_list = [int(file.split('_')[1].split('.')[0]) for file in os.listdir(ckpt_dir)]
	file_list = sorted(file_list)
	return 'ckpt_' + str(file_list[-1]) + '.pth.tar'

def update_lr(optimizer, num_batch, lr_init, lr_decay, lr_gamma):
	new_lr = lr_init * lr_gamma ** (num_batch / lr_decay)
	for param_group in optimizer.param_groups:
		param_group['lr'] = new_lr
	return new_lr

def unnormalize(data, mean_vec, std_vec, dim_use_vec):
	m = data.shape[0]
	n = mean_vec.shape[0]
	orig_data = np.zeros((m, n))
	orig_data[:, dim_use_vec] = data

	mean_vec = mean_vec.reshape((1, n))
	std_vec = std_vec.reshape((1, n))
	mean_mat = np.repeat(mean_vec, m, axis = 0)
	std_mat = np.repeat(std_vec, m, axis = 0)

	orig_data = np.multiply(orig_data, std_mat) + mean_mat

	return orig_data

def train(train_data, model, optimizer, criterion, start_epoch, end_epoch, snapshot,
	cur_lr, lr_init, lr_decay, lr_gamma, ckpt_dir, CUDA, max_norm = True):

	num_batch = 0			#to be modified
	cudnn.benchmark = True
	for epoch in range(start_epoch, end_epoch):
		print('-----------epoch: {}-----------'.format(epoch))
		model.train()

		for i, (data_2d, data_3d) in enumerate(train_data):
			num_batch += 1
			if num_batch % lr_decay == 0 or num_batch == 1:
				cur_lr = update_lr(optimizer, num_batch ,lr_init ,lr_decay, lr_gamma)

			if CUDA:
				input = Variable(data_2d.cuda()) if CUDA else Variable(data_2d.cpu())
				target = Variable(data_3d.cuda(async=True)) if CUDA else Variable(data_3d.cpu())

			optimizer.zero_grad()

			output = model(input)
			loss = criterion(output, target)
			print('batch_id: {} ,learning_rate: {} ,loss_train: {} '.format(i, cur_lr, loss))
			loss.backward()
			if max_norm:
				nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)
			optimizer.step()


		if epoch % snapshot == 0 and epoch != 0:
			print('---------Saving checkpoint---------')
			state = {
			'lr' : cur_lr,
			'epoch' : epoch,
			'model' : model.state_dict(),
			'optim' : optimizer.state_dict()
			}
			save_file = os.path.join(ckpt_dir ,'ckpt_{}.pth.tar'.format(epoch))
			torch.save(state, save_file)
			print('---------Saving complete---------')

def main():
	args = Options().parse()
	model = Model()
	# CUDA = torch.cuda.is_available()
	CUDA = False
	if CUDA:
		model = model.cuda()
	model.apply(weight_init)
	criterion = nn.MSELoss(size_average = True).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

	start_epoch = 0
	end_epoch = args.epochs
	cur_lr = args.lr
	lr_init = args.lr

	# train mode
	if args.train:
		train_data = DataLoader(dataset = Human36m(data_path = args.data_dir, is_train = True), batch_size = args.train_batch, 
			shuffle = False, num_workers = args.job, pin_memory = True)

		if args.resume:
			last_ckpt = sort_ckpt(args.ckpt)
			cktp_file = os.path.join(args.ckpt, last_ckpt)
			ckpt = torch.load(cktp_file)
			start_epoch = ckpt['epoch']
			cur_lr = ckpt['lr']
			model.load_state_dict(ckpt['model'])
			optimizer.load_state_dict(ckpt['optim'])
			print('Loaded checkpoint from {}'.format(last_ckpt))

		train(train_data, model, optimizer, criterion, start_epoch, end_epoch, args.snapshot,
			cur_lr, lr_init, args.lr_decay, args.lr_gamma, args.ckpt, CUDA)

	# test mode
	elif args.test:
		test_data = DataLoader(dataset = Human36m(data_path = args.data_dir, is_train = False), batch_size = args.test_batch,
			shuffle = True, num_workers = args.job, pin_memory = True)

		if not args.load:
			print('Need to specify model')
			sys.exit()

		cktp_file = args.load
		ckpt = torch.load(cktp_file)
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optim'])
		print('Loaded checkpoint from {}'.format(args.load))

		for i, (data, target) in enumerate(test_data):
			data = Variable(data.cuda()) if CUDA else Variable(data.cpu())
			target = Variable(target.cuda()) if CUDA else Variable(target.cpu())
			output = model(data)

			stat_3d = torch.load(os.path.join(args.data_dir, 'stat_3d.pth.tar'))
			stat_2d = torch.load(os.path.join(args.data_dir, 'stat_2d.pth.tar'))

			unnormal_output = unnormalize(output.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
			unnormal_target = unnormalize(target.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
			unnormal_data = unnormalize(data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])

			used_target = unnormal_target[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
			used_output = unnormal_output[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
			used_data = unnormal_data[:, stat_2d['dim_use']].reshape((-1, 2)).reshape((-1, 16, 2))

			target_3d = used_target[0]
			data_3d = used_output[0]
			data_2d = used_data[0]

			draw_picture(data_2d)
			draw_picture(target_3d)
			draw_picture(data_3d)


if __name__ == '__main__':
	main()
