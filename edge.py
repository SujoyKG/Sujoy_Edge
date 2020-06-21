import mahotas as mh
import pylab as pl
import numpy as np


def sujoy(img, kernel_nhood=0, just_filter=False):
	
	'''
	edges = sujoy(img, kernel_nhood=0, just_filter=False)
	Compute edges using Sujoy's algorithm
	`edges` is a binary image of edges computed according to Sujoy's algorithm.
	PAPER LINK: https://www.ijert.org/research/a-better-first-derivative-approach-for-edge-detection-IJERTV2IS110616.pdf
	
	Parameters
	----------
	img : Any 2D-ndarray
	kernel_nhood : 0(default) or 1
		if 0, kernel is based on 4-neighborhood
		else , kernel is based on 8-neighborhood
	just_filter : boolean, optional
		If true, then return the result of filtering the image with the Sujoy's
		filters, but do not threshold (default is False).
	Returns
	-------
	edges : ndarray
		Binary image of edges, unless `just_filter`, in which case it will be
		an array of floating point values.
	'''
	
	img = np.array(img, dtype=np.float)
	if img.ndim != 2:
		raise ValueError('mahotas.sujoy: Only available for 2-dimensional images')
	img -= img.min()
	ptp = img.ptp()
	if ptp == 0:
		return img
	img /= ptp


	if kernel_nhood:
		krnl_h = np.array([[0,-1,-1,-1,0],[0,-1,-1,-1,0],[0,0,0,0,0],[0,1,1,1,0],[0,1,1,1,0]])/12.
		krnl_v = np.array([[0,0,0,0,0],[-1,-1,0,1,1],[-1,-1,0,1,1],[-1,-1,0,1,1],[0,0,0,0,0]])/12.
	else:
		krnl_h = np.array([[0,0,-1,0,0],[0,-1,-1,-1,0],[0,0,0,0,0],[0,1,1,1,0],[0,0,1,0,0]])/8.
		krnl_v = np.array([[0,0,0,0,0],[0,-1,0,1,0],[-1,-1,0,1,1],[0,-1,0,1,0],[0,0,0,0,0]])/8.

	grad_h = mh.convolve(img, krnl_h, mode='nearest')
	grad_v = mh.convolve(img, krnl_v, mode='nearest')

	grad_h **=2
	grad_v **=2

	grad = grad_h
	grad += grad_v
	if just_filter:
		return grad
	t = np.sqrt(grad.mean())

	return mh.regmax(grad)*(np.sqrt(grad)>t)
