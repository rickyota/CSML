import pickle
import os

from .imclass import ImClass
from . import infer_im
from . import stats

# inference step


def infer_step(fname_infer, fname_save, fname_model,
               thre_discard, wid_dilate, fstats):

    print("Start inferring.")

    cim = ImClass('infer', fname_infer=fname_infer, fname_inferred=fname_save)

    if not os.path.isfile(fname_model):
        raise FileNotFoundError("Model file not found: {}.".format(fname_model))
    try:
        with open(fname_model, 'rb') as p:
            data_model = pickle.load(p)
    except Exception as e:
        raise Exception(e, "Cannot open model file: {}".format(fname_model))

    model_infer = data_model['model']
    cim.change_hgh_wid(data_model['shape'])
    print("Info of FCN Classifier: \n",
          "\t", "hgh,wid", data_model['shape'], "\n",
          "\t", "test dot acc", "{:.3f}".format(data_model['testacc']))

    # for i, (fname_infer, fname_inferred, fname_tag, fname_stats) \
    #        in enumerate(zip(cim.fnames_infer, cim.fnames_inferred, cim.fnames_tag, cim.fnames_stats)):
    for i in range(len(cim.fnames_infer)):
        fname_infer = cim.fnames_infer[i]
        fname_inferred = cim.fnames_inferred[i]
        fname_tag = cim.fnames_tag[i]
        fname_stats = cim.fnames_stats[i]

        if fname_infer.endswith('.tif') or fname_infer.endswith('.tiff'):
            ims_infer = cim.read_im_tif(fname_infer)
            ims_inferred = []
            for im_infer in ims_infer:
                im_inferred = infer_im.infer_im(
                    model_infer, cim.shape, im_infer, thre_discard, wid_dilate)
                ims_inferred.append(im_inferred)
            cim.save_im_tif(ims_inferred, fname_inferred)

            ims_tag = []
            dfs_stats = []
            for im_inferred in ims_inferred:
                im_tag, df_stats = stats.tag(im_inferred, fstats)
                ims_tag.append(im_tag)
                dfs_stats.append(df_stats)
            cim.save_im_tif_tag(ims_tag, fname_tag)
            if fstats:
                cim.save_xlsx(dfs_stats, fname_stats)

        else:
            im_infer = cim.read_im(fname_infer)
            im_inferred = infer_im.infer_im(
                model_infer, cim.shape, im_infer, thre_discard, wid_dilate)
            cim.save_im(im_inferred, fname_inferred)

            im_tag, df_stats = stats.tag(im_inferred, fstats)
            cim.save_im_tag(im_tag, fname_tag)
            if fstats:
                cim.save_xlsx(df_stats, fname_stats)

        if i % 5 == 4:
            print("Done inferring", i + 1, "/", len(cim.fnames_infer))

    print("Done inferring.")
