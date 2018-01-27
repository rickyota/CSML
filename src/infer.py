from imclass import ImClass
from processing import infer_imwhole

import pickle
import os

import traceback


# inferrence step
def infer_step(fname_infer="", fname_save="", fname_model="", thre_discard=1000, wid_dilate=1, thre_fill=1):

    print("start inferring.")

    im = ImClass('infer', fname_i=fname_infer)

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
    im.change_hgh_wid(data_model['shape'])
    print("test acc", "{:.3f}".format(data_model['testacc'][-1]))

    try:
        x_whole_inferred = infer_imwhole(
            model_infer, im, thre_discard, wid_dilate, thre_fill)
    except MemoryError as e:
        raise MemoryError(
            e, "Too many images to be inferred. Decrease number of images.")
    except Exception as e:
        print("Error!!!", e)
        traceback.print_exc()
    im.save_image(x_whole_inferred, fname_save)

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
