from imclass import ImClass
from processing import infer_imwhole_each

import pickle
import os

import traceback


# inferrence step
def infer_step(fname_infer="", fname_save="", fname_model="",
               thre_discard=1000, wid_dilate=1, thre_fill=1):

    print("start inferring.")

    cim = ImClass('infer', fname_i=fname_infer, fname_s=fname_save)

    if not os.path.isfile(fname_model):
        raise FileNotFoundError(
            "Model file not found: {}.".format(fname_model))
    try:
        with open(fname_model, 'rb') as p:
            data_model = pickle.load(p)
    except Exception as e:
        raise Exception(e, "Cannot open model file: {}".fotmat(fname_model))

    model_infer = data_model['model']
    print("info of FCN Classifier: ")
    print("hgh,wid", data_model['shape'])
    cim.change_hgh_wid(data_model['shape'])
    print("test acc", "{:.3f}".format(data_model['testacc'][-1]))

    """
    x_whole_inferred = infer_imwhole(
        model_infer, cim, thre_discard, wid_dilate, thre_fill)
    """
    if cim.type_infer == 'folder':
        num_im = len(cim.fnames_infer)
        for i in range(num_im):
            x_whole_each = cim.load_one_image(cim.fnames_infer[i])
            im_infer_each = infer_imwhole_each(
                model_infer, cim, x_whole_each, thre_discard, wid_dilate, thre_fill)
            cim.save_image(im_infer_each, cim.fnames_inferred[i], 'folder')

            if i % 5 == 4:
                print("Done inferring", i + 1, "/", num_im)

    elif cim.type_infer == 'file':
        num_im = cim.get_numframe()
        im_infer = []
        for i in range(num_im):
            x_whole_each = cim.load_xwhole(i)
            im_infer_each = infer_imwhole_each(
                model_infer, cim, x_whole_each, thre_discard, wid_dilate, thre_fill)
            im_infer.append(im_infer_each)
            if i % 5 == 4:
                print("Done inferring", i + 1, "/", num_im)
        cim.save_image(im_infer, fname_save, 'file')

    """
    # infer whole images
    def infer_imwhole(model, cim, thre_discard, wid_dilate, thre_fill):
    
        num_im = cim.get_numframe() if cim.type_infer == 'file' else len(cim.fnames_infer)
        im_infer = []
        for i in range(num_im):
            if cim.type_infer == 'file':
                x_whole_each = cim.load_xwhole(i)
            else:
                x_whole_each = cim.load_one_image(cim.fnames_infer[i])
            im_infer_each = infer_imwhole_each(
                model, cim, x_whole_each, thre_discard, wid_dilate, thre_fill)
            im_infer.append(im_infer_each)
            if i % 5 == 4:
                print("Done inferring", i + 1, "/", num_im)
    
    """

    """
    except MemoryError as e:
        raise MemoryError(
            e, "Too many images to be inferred. Decrease number of images.")
    """

    #cim.save_image(x_whole_inferred, fname_save)

    print("done infering.")


if __name__ == '__main__':
    # filenames for infer

    fname_infer = "../data/embryos_infer.tiff"
    fname_save = "../result/embryos_inferred.tiff"
    fname_model = "../data/model.pkl"

    # parameters for infer
    thre_discard = 1000
    wid_dilate = 1
    thre_fill = 1

    infer_step(fname_infer=fname_infer, fname_save=fname_save, fname_model=fname_model,
               thre_discard=thre_discard, wid_dilate=wid_dilate, thre_fill=thre_fill)
