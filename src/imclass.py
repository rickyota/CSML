import numpy as np
from cv2 import connectedComponents
from PIL import Image
import os
import warnings


# class of Images
class ImClass:

	def __init__(self, usetype, fname_i="", fname_x="", fname_t="",
				N_train=25000, N_test=3000, hgh=32, wid=32):
		
		# input size of classifier
		self.hgh = hgh
		self.wid = wid
		self.shapex = [1, self.hgh, self.wid]
		self.shapet = [self.hgh, self.wid]
		self.fnames = []
		
		if usetype == 'train':
			im_x, im_t = self.load_imx(fname_x), self.load_imt(fname_t)
			self.imdata_pkl = self.make_pkl(im_x, im_t, N_test, N_train)

		elif usetype == 'infer':
			if os.path.isfile(fname_i):
				self.type_infer = 'file'
				self.file_xwhole, self.numframe_xwhole = self.load_file(fname_i)
			elif os.path.isdir(fname_i):
				self.type_infer = 'folder'
				self.fnames_infer = self.get_listdir(fname_i)
				self.fnames_onlyname_infer = self.get_listdir_onlyname(fname_i)
				
			else: raise FileNotFoundError("file or folder not found.")
	
	# load images
	def load_imx(self, fname):
		if os.path.isfile(fname):
			im, num = self.load_file(fname)
			ims = []
			for i in range(num):
				im.seek(i)
				im_tmp = np.asarray(im.convert('L')) / 255.0
				ims.append(im_tmp)
		elif os.path.isdir(fname):
			self.fnames = self.get_listdir(fname)
			if not self.fnames: raise FileNotFoundError("no files in the folder: {0}.".format(fname))
			
			ims = []
			for filename in self.fnames:
				im_tmp = self.load_one_image(filename)
				ims.append(im_tmp)
		else: raise FileNotFoundError("file or folder not found.")
		
		ims = np.asarray(ims, np.float32)
		
		# if there is only one image
		if len(ims.shape) == 2:
			ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))
		
		return ims

	# load contoured images
	def load_imt(self, fname):
		if os.path.isfile(fname):
			im, num = self.load_file(fname)
			ims = []
			for i in range(num):
				im.seek(i)
				im_tmp = np.asarray(im.convert('L')) / 255
				ims.append(im_tmp)
		elif os.path.isdir(fname):
			self.fnames = self.get_listdir(fname)
			if not self.fnames: raise FileNotFoundError("no files in the folder: {0}.".format(fname))
			
			ims = []
			for filename in self.fnames:
				im_tmp = self.load_one_image(filename)
				ims.append(im_tmp)
		else:
			raise FileNotFoundError("file or folder not found.")
		
		ims = np.asarray(ims, np.int32)
		
		if len(ims.shape) == 2:
			ims = ims.reshape((-1, ims.shape[0], ims.shape[1]))
		
		return ims

	# load batches
	def load_batch(self):
		return self.imdata_pkl['x_train'], self.imdata_pkl['t_train'], self.imdata_pkl['x_test'], self.imdata_pkl['t_test']
	
	def get_listdir(self, fname):
		fnames = os.listdir(fname)
		fnames = list(filter(lambda f: f[0] != ".", fnames))
		fnames = [fname + "/" + filename for filename in fnames]
		return fnames
	
	def get_listdir_onlyname(self, fname):
		fnames = os.listdir(fname)
		fnames = list(filter(lambda f: f[0] != ".", fnames))
		return fnames
		
	# save image
	def save_image(self, ims, fname):
		ims = [Image.fromarray(im) for im in ims]
		if os.path.isfile(fname):
			if len(ims) == 1:
				ims[0].save(fname, save_all=True)
			else:
				ims[0].save(fname, save_all=True, append_images=ims[1:])
				
		else:
			if os.path.isdir(fname):
				warnings.warn("folder is being overwritten.")
			else:
				os.mkdir(fname)
		
			fnames = self.fnames_onlyname_infer
			fnames = [fname + "/" + filename for filename in fnames]
			for im, filename in zip(ims, fnames):
				im.save(filename)

	# load image from non-tiff file
	def load_one_image(self, fname):
		fim = Image.open(fname)
		im = np.asarray(fim.convert('L')) / 255.0
		return im
				
	# load file and number of frame
	def load_file(self, fname):
		im = Image.open(fname)
		num = 0
		try:
			while True:
				im.seek(num)
				num = num + 1
		except EOFError:
			pass
		
		return im, num
	
	# get number of frame
	def get_numframe(self):
		return self.numframe_xwhole

	# load num-th whole image 
	def load_xwhole(self, num):
		im = self.file_xwhole
		im.seek(num)
		im_xwhole = np.asarray(im.convert('L'), np.float32) / 255.0
		
		return im_xwhole
	
	# make file that all images packed in 
	def make_pkl(self, im_x, im_t, N_test, N_train):
		imdata_pkl = {}
		imdata_pkl['x_train'], imdata_pkl['t_train'], imdata_pkl['x_test'], imdata_pkl['t_test'] = self.pickimages(im_x, im_t, N_test, N_train)
		return imdata_pkl

	# choose images suitable for small training images
	def pickimages(self, im_x, im_t, N_test, N_train):
		N_each = int((N_train + N_test) / im_x.shape[0])
		x_tmp, t_tmp = [], []
		
		for i, (im_x_each, im_t_each) in enumerate(zip(im_x, im_t)):
			bound = self.getBound(im_t_each)
			xs, ys = np.where(bound == True)		
			perm = np.random.permutation(xs.shape[0])
			xs, ys = xs[perm[0:2 * N_each]], ys[perm[0:2 * N_each]]
			
			count = 0;
			for i in range(2 * N_each):
				x, y = xs[i], ys[i]
				im_t_tmp = im_t_each[x:x + self.hgh, y:y + self.wid]
				if self.isInBound(bound, x, y):
					im_x_tmp = im_x_each[x:x + self.hgh, y:y + self.wid]
					x_tmp.append(im_x_tmp), t_tmp.append(im_t_tmp)
					count = count + 1
					if count >= N_each: break;
					
		x_tmp, t_tmp = np.asarray(x_tmp, np.float32), np.asarray(t_tmp, np.int32)
		x_tmp, t_tmp = x_tmp.reshape([-1] + self.shapex), t_tmp.reshape([-1] + self.shapet)		
		
		x_train, t_train = x_tmp[0:N_train], t_tmp[0:N_train]
		x_test, t_test = x_tmp[N_train:N_train + N_test], t_tmp[N_train:N_train + N_test]
		
		return x_train, t_train, x_test, t_test
	
	# get criteria of whether suitable or not
	def getBound(self, im_t_each):
		_, bound = connectedComponents(np.uint8(1 - im_t_each))
		bound[bound != 1] = 0
		bound = 1 - bound
		bound = np.asarray(bound, bool)
		return bound
	
	# judge based on criteria
	def isInBound(self, bound, x, y):
		return bound[x + self.hgh, y] and bound[x, y + self.wid] and bound[x + self.hgh, y + self.wid]
		
