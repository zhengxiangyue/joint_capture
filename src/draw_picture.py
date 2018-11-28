#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_picture3(data_2d, data_3d, target):
	fig = plt.figure()

	ax1 = fig.add_subplot(131)
	ax2 = fig.add_subplot(132, projection='3d')
	ax3 = fig.add_subplot(133, projection='3d')

	ax1.scatter(data_2d[:, 0], data_2d[:, 1], c = 'b', marker = 'o')
	ax2.scatter(data_3d[:, 2], data_3d[:, 0], data_3d[:, 1], c = 'b', marker = 'o')
	ax3.scatter(target[:, 2], target[:, 0], target[:, 1], c = 'b', marker = 'o')

	ax1.plot([data_2d[0, 0], data_2d[1, 0], data_2d[2, 0], data_2d[3, 0]], [data_2d[0, 1], data_2d[1, 1], data_2d[2, 1], data_2d[3, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[0, 0], data_2d[4, 0], data_2d[5, 0], data_2d[6, 0]], [data_2d[0, 1], data_2d[4, 1], data_2d[5, 1], data_2d[6, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[0, 0], data_2d[7, 0], data_2d[8, 0], data_2d[9, 0]], [data_2d[0, 1], data_2d[7, 1], data_2d[8, 1], data_2d[9, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[8, 0], data_2d[10, 0], data_2d[11, 0], data_2d[12, 0]], [data_2d[8, 1], data_2d[10, 1], data_2d[11, 1], data_2d[12, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[8, 0], data_2d[13, 0], data_2d[14, 0], data_2d[15, 0]], [data_2d[8, 1], data_2d[13, 1], data_2d[14, 1], data_2d[15, 1]], c = 'r', linewidth=3.0)

	ax2.plot([data_3d[0, 2], data_3d[1, 2], data_3d[2, 2]], 
		[data_3d[0, 0], data_3d[1, 0], data_3d[2, 0]],
	 	[data_3d[0, 1], data_3d[1, 1], data_3d[2, 1]], c = 'r', linewidth = 3.0)	#leg
	ax2.plot([data_3d[3, 2], data_3d[4, 2], data_3d[5, 2]],
	 	[data_3d[3, 0], data_3d[4, 0], data_3d[5, 0]],
	 	[data_3d[3, 1], data_3d[4, 1], data_3d[5, 1]], c = 'r', linewidth = 3.0)	#leg
	ax2.plot([data_3d[6, 2], data_3d[7, 2], data_3d[8, 2], data_3d[9, 2]],
 		[data_3d[6, 0], data_3d[7, 0], data_3d[8, 0], data_3d[9, 0]],
	 	[data_3d[6, 1], data_3d[7, 1], data_3d[8, 1], data_3d[9, 1]], c = 'r', linewidth = 3.0)	#body
	ax2.plot([data_3d[8, 2], data_3d[10, 2], data_3d[11, 2], data_3d[12, 2]],
		[data_3d[8, 0], data_3d[10, 0], data_3d[11, 0], data_3d[12, 0]],
		[data_3d[8, 1], data_3d[10, 1], data_3d[11, 1], data_3d[12, 1]], c = 'r', linewidth = 3.0)
	ax2.plot([data_3d[8, 2], data_3d[13, 2], data_3d[14, 2], data_3d[15, 2]],
	 	[data_3d[8, 0], data_3d[13, 0], data_3d[14, 0], data_3d[15, 0]],
	  	[data_3d[8, 1], data_3d[13, 1], data_3d[14, 1], data_3d[15, 1]], c = 'r', linewidth = 3.0)
	ax2.plot([data_3d[0, 2], data_3d[6, 2], data_3d[3, 2]],
	 	[data_3d[0, 0], data_3d[6, 0], data_3d[3, 0]],
	  	[data_3d[0, 1], data_3d[6, 1], data_3d[3, 1]], c = 'r', linewidth = 3.0)

	ax3.plot([target[0, 2], target[1, 2], target[2, 2]], 
		[target[0, 0], target[1, 0], target[2, 0]],
	 	[target[0, 1], target[1, 1], target[2, 1]], c = 'r', linewidth = 3.0)	#leg
	ax3.plot([target[3, 2], target[4, 2], target[5, 2]],
	 	[target[3, 0], target[4, 0], target[5, 0]],
	 	[target[3, 1], target[4, 1], target[5, 1]], c = 'r', linewidth = 3.0)	#leg
	ax3.plot([target[6, 2], target[7, 2], target[8, 2], target[9, 2]],
 		[target[6, 0], target[7, 0], target[8, 0], target[9, 0]],
	 	[target[6, 1], target[7, 1], target[8, 1], target[9, 1]], c = 'r', linewidth = 3.0)	#body
	ax3.plot([target[8, 2], target[10, 2], target[11, 2], target[12, 2]],
		[target[8, 0], target[10, 0], target[11, 0], target[12, 0]],
		[target[8, 1], target[10, 1], target[11, 1], target[12, 1]], c = 'r', linewidth = 3.0)
	ax3.plot([target[8, 2], target[13, 2], target[14, 2], target[15, 2]],
	 	[target[8, 0], target[13, 0], target[14, 0], target[15, 0]],
	  	[target[8, 1], target[13, 1], target[14, 1], target[15, 1]], c = 'r', linewidth = 3.0)
	ax3.plot([target[0, 2], target[6, 2], target[3, 2]],
	 	[target[0, 0], target[6, 0], target[3, 0]],
	  	[target[0, 1], target[6, 1], target[3, 1]], c = 'r', linewidth = 3.0)

	ax1.set_title('2d input')
	ax2.set_title('prediction')
	ax3.set_title('target')
	ax1.invert_yaxis()
	ax2.invert_zaxis()
	ax3.invert_zaxis()

	plt.show()

def draw_picture2(data_2d, data_3d):
	fig = plt.figure()

	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122, projection='3d')

	ax1.scatter(data_2d[:, 0], data_2d[:, 1], c = 'b', marker = 'o')
	ax2.scatter(data_3d[:, 2], data_3d[:, 0], data_3d[:, 1], c = 'b', marker = 'o')

	ax1.plot([data_2d[0, 0], data_2d[1, 0], data_2d[2, 0], data_2d[3, 0]], [data_2d[0, 1], data_2d[1, 1], data_2d[2, 1], data_2d[3, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[0, 0], data_2d[4, 0], data_2d[5, 0], data_2d[6, 0]], [data_2d[0, 1], data_2d[4, 1], data_2d[5, 1], data_2d[6, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[0, 0], data_2d[7, 0], data_2d[8, 0], data_2d[9, 0]], [data_2d[0, 1], data_2d[7, 1], data_2d[8, 1], data_2d[9, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[8, 0], data_2d[10, 0], data_2d[11, 0], data_2d[12, 0]], [data_2d[8, 1], data_2d[10, 1], data_2d[11, 1], data_2d[12, 1]], c = 'r', linewidth=3.0)
	ax1.plot([data_2d[8, 0], data_2d[13, 0], data_2d[14, 0], data_2d[15, 0]], [data_2d[8, 1], data_2d[13, 1], data_2d[14, 1], data_2d[15, 1]], c = 'r', linewidth=3.0)

	ax2.plot([data_3d[0, 2], data_3d[1, 2], data_3d[2, 2]], 
		[data_3d[0, 0], data_3d[1, 0], data_3d[2, 0]],
	 	[data_3d[0, 1], data_3d[1, 1], data_3d[2, 1]], c = 'r', linewidth = 3.0)	#leg
	ax2.plot([data_3d[3, 2], data_3d[4, 2], data_3d[5, 2]],
	 	[data_3d[3, 0], data_3d[4, 0], data_3d[5, 0]],
	 	[data_3d[3, 1], data_3d[4, 1], data_3d[5, 1]], c = 'r', linewidth = 3.0)	#leg
	ax2.plot([data_3d[6, 2], data_3d[7, 2], data_3d[8, 2], data_3d[9, 2]],
 		[data_3d[6, 0], data_3d[7, 0], data_3d[8, 0], data_3d[9, 0]],
	 	[data_3d[6, 1], data_3d[7, 1], data_3d[8, 1], data_3d[9, 1]], c = 'r', linewidth = 3.0)	#body
	ax2.plot([data_3d[8, 2], data_3d[10, 2], data_3d[11, 2], data_3d[12, 2]],
		[data_3d[8, 0], data_3d[10, 0], data_3d[11, 0], data_3d[12, 0]],
		[data_3d[8, 1], data_3d[10, 1], data_3d[11, 1], data_3d[12, 1]], c = 'r', linewidth = 3.0)
	ax2.plot([data_3d[8, 2], data_3d[13, 2], data_3d[14, 2], data_3d[15, 2]],
	 	[data_3d[8, 0], data_3d[13, 0], data_3d[14, 0], data_3d[15, 0]],
	  	[data_3d[8, 1], data_3d[13, 1], data_3d[14, 1], data_3d[15, 1]], c = 'r', linewidth = 3.0)
	ax2.plot([data_3d[0, 2], data_3d[6, 2], data_3d[3, 2]],
	 	[data_3d[0, 0], data_3d[6, 0], data_3d[3, 0]],
	  	[data_3d[0, 1], data_3d[6, 1], data_3d[3, 1]], c = 'r', linewidth = 3.0)

	ax1.set_title('2d input')
	ax2.set_title('prediction')
	ax1.invert_yaxis()
	ax2.invert_zaxis()

	plt.show()