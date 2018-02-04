from imclass import ImClass
from processing import infer_im

import pickle
import os


# inferrence step
def infer_step(fname_infer="", fname_save="", fname_model="",
               thre_discard=1000, wid_dilate=1, thre_fill=1):

    print("start inferring.")

    cim = ImClass('infer', fname_infer=fname_infer, fname_inferred=fname_save)

    if not os.path.isfile(fname_model):
        raise FileNotFoundError(
            "Model file not found: {}.".format(fname_model))
    try:
        with open(fname_model, 'rb') as p:
            data_model = pickle.load(p)
    except Exception as e:
        raise Exception(e, "Cannot open model file: {}".fotmat(fname_model))

    model_infer = data_model['model']
    cim.change_hgh_wid(data_model['shape'])
    print("Info of FCN Classifier: \n",
          "\t", "hgh,wid", data_model['shape'], "\n",
          "\t", "test acc", "{:.3f}".format(data_model['testacc'][-1]))

    if cim.type_infer == 'folder':
        num_im = len(cim.fnames_infer)
        for i in range(num_im):
            im_infer = cim.read_im_folder(cim.fnames_infer[i])
            im_inferred = infer_im(
                model_infer, cim, im_infer, thre_discard, wid_dilate, thre_fill)
            cim.save_image(im_inferred, cim.fnames_inferred[i])

            if i % 5 == 4:
                print("Done inferring", i + 1, "/", num_im)

    elif cim.type_infer == 'file':
        num_im = cim.num_infer
        ims_inferred = []
        try:
            for i in range(num_im):
                im_infer = cim.read_im_file(i)
                im_inferred = infer_im(
                    model_infer, cim, im_infer, thre_discard, wid_dilate, thre_fill)
                ims_inferred.append(im_inferred)
                if i % 5 == 4:
                    print("Done inferring", i + 1, "/", num_im)
            cim.save_image(ims_inferred, fname_save)
        except MemoryError as e:
            raise MemoryError(
                e, "Too many images to be inferred. Decrease number of images.")

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
