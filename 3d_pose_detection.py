#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import numpy as np
from src.dataset import Human36m
from src.opt import *
from src.model2 import Model, weight_init
# from src.model import LinearModel, weight_init
from src.draw_picture import draw_picture3, draw_picture2
from src.utils import sort_ckpt, update_lr, unnormalize, normalize, loss_Average
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def train(train_data, test_data, model, optimizer, criterion, start_epoch, end_epoch, num_batch, snapshot,
	cur_lr, lr_init, lr_decay, lr_gamma, ckpt_dir, stat_data_dir, log_dir, CUDA, max_norm = True):

	# num_batch = 1242921		# if resume, load from ckpt
	cudnn.benchmark = True
	fw = open(os.path.join(log_dir, 'log.txt'), 'a')
	for epoch in range(start_epoch + 1, end_epoch + 1):
		print('-----------epoch: {}-----------'.format(epoch))
		loss_avg = loss_Average()
		print('loss average is : {}'.format(loss_avg.avg))
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
			loss_avg.update(loss.item(), data_2d.size(0))
			print('batch_id: {} ,learning_rate: {} ,loss_train: {} '.format(i, cur_lr, loss_avg.avg))
			loss.backward()

			if max_norm:
				nn.utils.clip_grad_norm(model.parameters(), max_norm = 1)
			optimizer.step()

		print('---------- Testing ---------')

		model.eval()
		test_loss, cord_err = test(test_data, model, optimizer, criterion, stat_data_dir, CUDA)

		print('---------- Test completed ---------')
		print('test loss: {}(std), corordinates error: {} '.format(test_loss, cord_err))

		content_to_write = str(epoch) + ',' + str(cur_lr) + ',' + str(loss_avg.avg) + ',' + str(test_loss) + ',' + str(cord_err) + '\n'
		fw.write(content_to_write)

		if epoch % snapshot == 0 and epoch != 0:
			print('---------Saving checkpoint---------')
			state = {
			'lr' : cur_lr,
			'epoch' : epoch,
			'num_batch' : num_batch,
			'model' : model.state_dict(),
			'optim' : optimizer.state_dict()
			}
			save_file = os.path.join(ckpt_dir ,'ckpt_{}.pth.tar'.format(epoch))
			torch.save(state, save_file)
			print('---------Saving complete---------')

	fw.close()

def test(test_data, model, optimizer, criterion, stat_data_dir, CUDA):

	loss_avg = loss_Average()
	cord_err = []
	for i,(data, target) in enumerate(test_data):
		data = Variable(data.cuda()) if CUDA else Variable(data.cpu())
		target = Variable(target.cuda()) if CUDA else Variable(target.cpu())
		output = model(data)

		loss = criterion(output, target)
		loss_avg.update(loss.item(), output.size(0))

		stat_3d = torch.load(os.path.join(stat_data_dir, 'stat_3d.pth.tar'))

		unnormal_output = unnormalize(output.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
		unnormal_target = unnormalize(target.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

		used_output = unnormal_output[:, stat_3d['dim_use']]
		used_target = unnormal_target[:, stat_3d['dim_use']]

		cord_dist = (used_output - used_target) ** 2

		index = 0
		distance = np.zeros((used_output.shape[0], 16))
		for i in np.arange(0, 16 * 3, 3):
			distance[:, index] = np.sqrt(np.sum(cord_dist[:, i : i + 3], axis = 1))
			index += 1

		cord_err.append(distance)

	cord_err = np.vstack(cord_err)
	cord_err = np.mean(cord_err)

	global best_cord_err
	if cord_err < best_cord_err:
		best_cord_err = cord_err
		best_state = {
		'best_cord_err' : cord_err,
		'model' : model.state_dict(),
		'optim' : optimizer.state_dict()
		}		
		torch.save(best_state, './best.pth.tar')

	return loss_avg.avg, cord_err


def main():

	args = Options().parse()
	model = Model()
	# model = LinearModel()
	CUDA = torch.cuda.is_available()
	if CUDA:
		model = model.cuda()
	model.apply(weight_init)
	criterion = nn.MSELoss(size_average = True).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

	start_epoch = 0
	end_epoch = args.epochs
	num_batch = 0
	cur_lr = args.lr
	lr_init = args.lr

	test_data = DataLoader(dataset = Human36m(data_path = args.data_dir, is_train = False), batch_size = args.test_batch,
		shuffle = True, num_workers = args.job, pin_memory = True)

	# train mode
	if args.train:
		train_data = DataLoader(dataset = Human36m(data_path = args.data_dir, is_train = True), batch_size = args.train_batch, 
			shuffle = False, num_workers = args.job, pin_memory = True)

		if args.resume:
			best_ckpt = torch.load('./best.pth.tar')
			global best_cord_err
			best_cord_err = best_ckpt['best_cord_err']
			last_ckpt = sort_ckpt(args.ckpt)
			cktp_file = os.path.join(args.ckpt, last_ckpt)
			ckpt = torch.load(cktp_file)
			start_epoch = ckpt['epoch']
			cur_lr = ckpt['lr']
			num_batch = ckpt['num_batch']
			model.load_state_dict(ckpt['model'])
			optimizer.load_state_dict(ckpt['optim'])
			print('Loaded checkpoint from {}'.format(last_ckpt))

		train(train_data, test_data, model, optimizer, criterion, start_epoch, end_epoch, num_batch, args.snapshot,
			cur_lr, lr_init, args.lr_decay, args.lr_gamma, args.ckpt, args.data_dir, args.log_dir, CUDA)

	# test mode
	elif args.test:
		if not args.load:
			print('Need to specify model')
			sys.exit()

		cktp_file = args.load
		ckpt = torch.load(cktp_file, map_location=lambda storage, loc: storage)
		model.load_state_dict(ckpt['state_dict'])
		# optimizer.load_state_dict(ckpt['optim'])
		print('Loaded checkpoint from {}'.format(args.load))

		model.eval()

		if not args.test_data:
			print('Need to specify input')
			sys.exit()

		data_file = args.test_data
		data = [float(num) for num in open(data_file).readline().split(',')]
		data = np.array(data).reshape((1, -1))

		stat_2d = torch.load(os.path.join(args.data_dir, 'stat_2d.pth.tar'))
		used_data = normalize(data, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])

		used_data = torch.from_numpy(used_data).float()

		used_data = Variable(used_data.cuda()) if CUDA else Variable(used_data.cpu())
		output = model(used_data)

		stat_3d = torch.load(os.path.join(args.data_dir, 'stat_3d.pth.tar'))
		unnormal_output = unnormalize(output.data.cpu(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
		used_output = unnormal_output[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
		
		output_3d = used_output[0]
		data = data.reshape((-1, 2)).reshape((-1, 16, 2))

		draw_picture2(data[0], output_3d)

		# for i, (data, target) in enumerate(test_data):

		# 	stat_3d = torch.load(os.path.join(args.data_dir, 'stat_3d.pth.tar'))
		# 	stat_2d = torch.load(os.path.join(args.data_dir, 'stat_2d.pth.tar'))

		# 	print data[0]
		# 	data = Variable(data.cuda()) if CUDA else Variable(data.cpu())
		# 	target = Variable(target.cuda()) if CUDA else Variable(target.cpu())
		# 	print data.size()
		# 	output = model(data)


		# 	unnormal_output = unnormalize(output.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
		# 	unnormal_target = unnormalize(target.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
		# 	unnormal_data = unnormalize(data.cpu().numpy(), stat_2d['mean'], stat_2d['std'], stat_2d['dim_use'])

		# 	used_target = unnormal_target[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
		# 	used_output = unnormal_output[:, stat_3d['dim_use']].reshape((-1, 3)).reshape((-1, 16, 3))
		# 	used_data = unnormal_data[:, stat_2d['dim_use']].reshape((-1, 2)).reshape((-1, 16, 2))

		# 	print unnormal_data[:, stat_2d['dim_use']][0]

		# 	target_3d = used_target[0]
		# 	data_3d = used_output[0]
		# 	data_2d = used_data[0]

		# 	# draw_picture3(data_2d, data_3d, target_3d)
		# 	draw_picture2(data_2d, data_3d)


best_cord_err = np.inf			#if resume, load from best ckpt

if __name__ == '__main__':
	main()
