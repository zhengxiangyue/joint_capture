#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_picture(data):
	fig = plt.figure()
	if data.shape[1] == 2:
		ax = fig.add_subplot(111)
		ax.scatter(data[:, 0], data[:, 1], c = 'b', marker = 'o')

		ax.plot([data[0, 0], data[1, 0], data[2, 0], data[3, 0]], [data[0, 1], data[1, 1], data[2, 1], data[3, 1]], c = 'r', linewidth=3.0)
		ax.plot([data[0, 0], data[4, 0], data[5, 0], data[6, 0]], [data[0, 1], data[4, 1], data[5, 1], data[6, 1]], c = 'r', linewidth=3.0)
		ax.plot([data[0, 0], data[7, 0], data[8, 0], data[9, 0]], [data[0, 1], data[7, 1], data[8, 1], data[9, 1]], c = 'r', linewidth=3.0)
		ax.plot([data[8, 0], data[10, 0], data[11, 0], data[12, 0]], [data[8, 1], data[10, 1], data[11, 1], data[12, 1]], c = 'r', linewidth=3.0)
		ax.plot([data[8, 0], data[13, 0], data[14, 0], data[15, 0]], [data[8, 1], data[13, 1], data[14, 1], data[15, 1]], c = 'r', linewidth=3.0)
		ax = plt.gca()
		ax.xaxis.set_ticks_position('top')
		ax.invert_yaxis()
	else:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(data[:, 2], data[:, 1], data[:, 0], c = 'b', marker = 'o')

		ax.plot([data[0, 2], data[1, 2], data[2, 2]], [data[0, 1], data[1, 1], data[2, 1]], [data[0, 0], data[1, 0], data[2, 0]], c = 'r', linewidth = 3.0)	#leg
		ax.plot([data[3, 2], data[4, 2], data[5, 2]], [data[3, 1], data[4, 1], data[5, 1]], [data[3, 0], data[4, 0], data[5, 0]], c = 'r', linewidth = 3.0)	#leg
		ax.plot([data[6, 2], data[7, 2], data[8, 2], data[9, 2]], [data[6, 1], data[7, 1], data[8, 1], data[9, 1]], [data[6, 0], data[7, 0], data[8, 0], data[9, 0]], c = 'r', linewidth = 3.0)	#body
		ax.plot([data[8, 2], data[10, 2], data[11, 2], data[12, 2]], [data[8, 1], data[10, 1], data[11, 1], data[12, 1]], [data[8, 0], data[10, 0], data[11, 0], data[12, 0]], c = 'r', linewidth = 3.0)
		ax.plot([data[8, 2], data[13, 2], data[14, 2], data[15, 2]], [data[8, 1], data[13, 1], data[14, 1], data[15, 1]], [data[8, 0], data[13, 0], data[14, 0], data[15, 0]], c = 'r', linewidth = 3.0)
		ax.plot([data[0, 2], data[6, 2], data[3, 2]], [data[0, 1], data[6, 1], data[3, 1]], [data[0, 0], data[6, 0], data[3, 0]], c = 'r', linewidth = 3.0)
		# ax.plot([data[0, 0], data[1, 0], data[2, 0], data[3, 0]], [data[0, 1], data[1, 1], data[2, 1], data[3, 1]],
		#  [data[0, 2], data[1, 2], data[2, 2], data[3, 2]], c = 'r', linewidth=3.0)
		# ax.plot([data[0, 0], data[4, 0], data[5, 0], data[6, 0]], [data[0, 1], data[4, 1], data[5, 1], data[6, 1]],
		#  [data[0, 2], data[4, 2], data[5, 2], data[6, 2]], c = 'r', linewidth=3.0)
		# ax.plot([data[0, 0], data[7, 0], data[8, 0], data[9, 0]], [data[0, 1], data[7, 1], data[8, 1], data[9, 1]],
		#  [data[0, 2], data[7, 2], data[8, 2], data[9, 2]], c = 'r', linewidth=3.0)
		# ax.plot([data[8, 0], data[10, 0], data[11, 0], data[12, 0]], [data[8, 1], data[10, 1], data[11, 1], data[12, 1]],
		#  [data[8, 2], data[10, 2], data[11, 2], data[12, 2]], c = 'r', linewidth=3.0)
		# ax.plot([data[8, 0], data[13, 0], data[14, 0], data[15, 0]], [data[8, 1], data[13, 1], data[14, 1], data[15, 1]],
		#  [data[8, 2], data[13, 2], data[14, 2], data[15, 2]], c = 'r', linewidth=3.0)

	plt.show()