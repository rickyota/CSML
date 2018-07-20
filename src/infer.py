from imclass import ImClass
import pickle
import os
from infer_im import infer_im


# inferrence step
def infer_step(fname_infer, fname_save, fname_model,
               thre_discard, wid_dilate):

    print("Start inferring.")

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
          "\t", "test acc", "{:.3f}".format(data_model['testacc']))

    num_im = len(cim.fnames_infer)
    for i in range(num_im):
        im_infer = cim.read_im(cim.fnames_infer[i])
        im_inferred = infer_im(
            model_infer, cim.shape, im_infer, thre_discard, wid_dilate)
        cim.save_im(im_inferred, cim.fnames_inferred[i])

        if i % 5 == 4:
            print("Done inferring", i + 1, "/", num_im)

    print("Done inferring.")
