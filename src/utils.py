#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt


class loss_Average:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def normalize(data, mean_vec, std_vec, dim_use_vec):

	mean_vec = mean_vec[dim_use_vec].reshape((1, -1))
	std_vec = std_vec[dim_use_vec].reshape((1, -1))

	data = (data - mean_vec) / std_vec

	return data 

def draw_leancurve(log_file):
	fr = open(log_file)
	fr.readline()
	lr = []; train_loss = []; test_loss = []; cord_err = []
	for record in fr.readlines():
		record = record.split(',')
		lr.append(float(record[1]))
		train_loss.append(float(record[2]))
		test_loss.append(float(record[3]))
		cord_err.append(float(record[4]))
	return lr, train_loss, test_loss, cord_err
