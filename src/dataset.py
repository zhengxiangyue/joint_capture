#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import os
import torch
from torch.utils.data import Dataset

class Human36m(Dataset):
	def __init__(self, data_path, is_train=True):
		self.data_path = data_path
		self.is_train = is_train
		self.train_input, self.train_output, self.test_input, self.test_output = [], [], [], []

		if self.is_train:
			input_file = os.path.join(self.data_path, 'train_2d.pth.tar')
			output_file = os.path.join(self.data_path, 'train_3d.pth.tar')
			self.train_2d = torch.load(input_file)
			self.train_3d = torch.load(output_file)

			for k2d in self.train_2d.keys():
				(sub, act, fname) = k2d
				k3d = k2d
				k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
				assert self.train_2d[k2d].shape[0] == self.train_3d[k3d].shape[0], 'training input & output not match'
				num_record = self.train_2d[k2d].shape[0]
				for i in range(num_record):
					self.train_input.append(self.train_2d[k2d][i])
					self.train_output.append(self.train_3d[k3d][i])

		else:
			input_file = os.path.join(self.data_path, 'test_2d.pth.tar')
			output_file = os.path.join(self.data_path, 'test_3d.pth.tar')
			self.test_2d = torch.load(input_file)
			self.test_3d = torch.load(output_file)

			for k2d in self.test_2d.keys():
				(sub, act, fname) = k2d
				k3d = k2d
				k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
				assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], 'testing input & output not match'
				num_record = self.test_2d[k2d].shape[0]
				for i in range(num_record):
					self.test_input.append(self.test_2d[k2d][i])
					self.test_output.append(self.test_3d[k3d][i])

	def __getitem__(self, index):
		if self.is_train:
			inputs = torch.from_numpy(self.train_input[index]).float()
			outputs = torch.from_numpy(self.train_output[index]).float()

		else:
			inputs = torch.from_numpy(self.test_input[index]).float()
			outputs = torch.from_numpy(self.test_output[index]).float()

		return inputs, outputs

	def __len__(self):
		if self.is_train:
			return len(self.train_input)
		else:
			return len(self.test_input)
