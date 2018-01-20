from imclass import ImClass
from processing import infer_imwhole, save_image

import pickle


# inferrence step
def infer_step(fname_infer="", fname_save="", fname_model="", thre_discard=1000, wid_dilate=1, thre_fill=1):
	
	print("start inferring.")
	
	im = ImClass('infer', fname_i=fname_infer)
	x_whole = im.load_xwhole()
	
	with open(fname_model, 'rb') as p:
		data_model = pickle.load(p)
	model_infer = data_model['model']
	print("info of FCN Classifier: ")
	print("hgh,wid", data_model['shape'])
	im.hgh, im.wid = data_model['shape']
	print("test acc", "{:.3f}".format(data_model['testacc'][-1]))
	
	x_whole_inferred = infer_imwhole(model_infer, im, x_whole, thre_discard, wid_dilate, thre_fill)
	
	save_image(x_whole_inferred, fname_save)  
	
	print("done infering.")
	
	
if __name__ == '__main__':
	# filenames for infer
	"""
	fname_infer = "../data/Cell_infer.tiff"
	fname_save = "../result/Cell_inferred.tiff"
	fname_model = "../data/model.pkl"
	"""
	
	fname_infer = "../data/180112a.tiff"
	fname_save = "../result/180112a_segmented.tiff"
	fname_model = "../data/model.pkl"
	
	# parameters for infer
	thre_discard = 1000
	wid_dilate = 1
	thre_fill = 1
	
	infer_step(fname_infer=fname_infer, fname_save=fname_save, fname_model=fname_model, thre_discard=thre_discard, wid_dilate=wid_dilate, thre_fill=thre_fill)
