from train import train_step
from infer import infer_step


def Cell_Segmentation():
    # 'both' or 'infer
    segtype = 'both'

    if segtype == 'both':
        # filenames for train
        fname_train = "../data/embryos_train.tiff"  # image
        fname_label = "../data/embryos_label.tiff"  # label
        fname_model = "../data/model.pkl"

        # filenames for infer
        fname_infer = "../data/embryos_infer.tiff"
        fname_save = "../result/embryos_inferred.tiff"

        # parameters for train
        N_test = 3000
        N_train = 25000
        N_epoch = 1
        batchsize = 100
        hgh = 32
        wid = 32

        # parameters for infer
        thre_discard = 1000
        wid_dilate = 1
        thre_fill = 1

        train_step(fname_train=fname_train, fname_label=fname_label, fname_model=fname_model,
                   N_test=N_test, N_train=N_train, N_epoch=N_epoch, batchsize=batchsize,
                   hgh=hgh, wid=wid)
        infer_step(fname_infer=fname_infer, fname_save=fname_save, fname_model=fname_model,
                   thre_discard=thre_discard, wid_dilate=wid_dilate, thre_fill=thre_fill)
        print("all done.")

    elif segtype == 'infer':
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
        print("all done.")


if __name__ == '__main__':
    Cell_Segmentation()
