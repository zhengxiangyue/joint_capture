#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import torch
import torch.nn as nn

def weight_init(x):
	if isinstance(x, nn.Linear):
		nn.init.kaiming_normal(x.weight)

class LinearUnit(nn.Module):
	def __init__(self, linear_size, p_dropout = 0.5):
		super(LinearUnit, self).__init__()
		self.linear_size = linear_size
		self.p_dropout = p_dropout
		#layers
		self.linear = nn.Linear(linear_size, linear_size)
		self.batch_norm = nn.BatchNorm1d(self.linear_size)
		self.drop_out = nn.Dropout(self.p_dropout)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		#stage 1
		y = self.linear(x)
		y = self.batch_norm(y)
		y = self.relu(y)
		y = self.drop_out(y)
		#stage 2
		y = self.linear(y)
		y = self.batch_norm(y)
		y = self.relu(y)
		y = self.drop_out(y)

		y = x + y

		return y

class Model(nn.Module):
	def __init__(self, linear_size = 1024, p_dropout = 0.5, num_stage = 2):
		super(Model, self).__init__()
		self.linear_size = linear_size
		self.p_dropout = p_dropout
		self.num_stage = num_stage
		self.inp_size = 16 * 2
		self.out_size = 16 * 3
		#layers
		self.linear1 = nn.Linear(self.inp_size, self.linear_size)
		self.linear2 = nn.Linear(self.linear_size, self.out_size)
		self.batch_norm = nn.BatchNorm1d(self.linear_size)
		self.relu = nn.ReLU(inplace=True)
		self.drop_out = nn.Dropout(self.p_dropout)
		self.linearList = []
		for i in range(self.num_stage):
			self.linearList.append(LinearUnit(self.linear_size, self.p_dropout))
		self.linearList = nn.ModuleList(self.linearList)

	def forward(self, x):
		y = self.linear1(x)
		y = self.batch_norm(y)
		y = self.relu(y)
		y = self.drop_out(y)

		for i in range(self.num_stage):
			y = self.linearList[i](y)

		y = self.linear2(y)

		return y