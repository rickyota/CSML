from train import train_step
from infer import infer_step


import os
from datetime import datetime
import argparse


def Cell_Segmentation():

    print("Start Cell Segmentation.")
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

    args = argument()

    if sys.argv[1].startswith('train_infer') and sys.argv[1].endswith('.ini'):
        print("Train and infer.")
        fname_train = "data/" + config['paths']['filename train']
        fname_label = "data/" + config['paths']['filename label']
        fname_model = "data/" + config['paths']['filename model']
        fname_infer = "data/" + config['paths']['filename infer']
        fname_save = "result/" + config['paths']['filename save']

        N_train = int(config['training parameters']['number train'])
        N_test = int(config['training parameters']['number test'])
        N_epoch = int(config['default']['number epoch'])
        batchsize = int(config['default']['number batch'])
        hgh = int(config['training parameters']['height'])
        wid = int(config['training parameters']['width'])

        thre_discard = int(config['inference parameters']['threshold discard'])
        wid_dilate = int(config['inference parameters']['width dilate'])
        thre_fill = int(config['inference parameters']['threshold fill'])

        try:
            train_step(fname_train=fname_train, fname_label=fname_label, fname_model=fname_model,
                       N_test=N_test, N_train=N_train, N_epoch=N_epoch, batchsize=batchsize, hgh=hgh, wid=wid)
        except Exception as e:
            raise Exception(e, "Got an error in training step.")
        try:
            infer_step(fname_infer=fname_infer, fname_save=fname_save, fname_model=fname_model,
                       thre_discard=thre_discard, wid_dilate=wid_dilate, thre_fill=thre_fill)
        except Exception as e:
            raise Exception(e, "Got an error in inference step.")
        print("All done.")

    elif sys.argv[1].startswith('infer') and sys.argv[1].endswith('.ini'):
        print("Only Infer.")
        fname_model = "data/" + config['paths']['filename model']
        fname_infer = "data/" + config['paths']['filename infer']
        fname_save = "result/" + config['paths']['filename save']

        thre_discard = int(config['inference parameters']['threshold discard'])
        wid_dilate = int(config['inference parameters']['width dilate'])
        thre_fill = int(config['inference parameters']['threshold fill'])

        try:
            infer_step(fname_infer=fname_infer, fname_save=fname_save, fname_model=fname_model,
                       thre_discard=thre_discard, wid_dilate=wid_dilate, thre_fill=thre_fill)
        except Exception as e:
            raise Exception(e, "Got an error in inference step.")
        print("All done.")


def argument():
    parser = argparse.ArgumentParser(
        prog='CSML',
        usage='Cell segmentation',
        description='description',
        epilog='description end',
        add_help=True,
    )

    parser.add_argument('-f', '--flaginfer',
                        help='Only inferrence',
                        action='store_true')
    parser.add_argument('-t', '--train', help='train folder name')
    parser.add_argument('-l', '--label', help='label folder name')
    parser.add_argument('-i', '--infer', help='infer folder name',
                        default='research_infer')
    parser.add_argument('-o', '--output', help='output folder name')
    parser.add_argument('-m', '--model', help='model file name')

    parser.add_argument('-nr', '--ntrain', help='number of training images',
                        type=int, default=50000)
    parser.add_argument('-ns', '--ntest', help='number of testing images',
                        type=int, default=3000)
    parser.add_argument('-he', '--height', help='height',
                        type=int, default=64)
    parser.add_argument('-wi', '--width', help='width',
                        type=int, default=64)
    parser.add_argument('-td', '--discard', help='threshold discard',
                        type=int, default=100)
    parser.add_argument('-wd', '--dilate', help='width dilation',
                        type=int, default=1)
    parser.add_argument('-ne', '--nepoch', help='number of epoch',
                        type=int, default=1)
    parser.add_argument('-nb', '--nbatch', help='number of batch',
                        type=int, default=100)
    parser.add_argument('-mo', '--mode', help='way to choose train images',
                        default='back')
    parser.add_argument('-in', '--interim', help='flag of save interm accuracy',
                        action='store_true')
    parser.add_argument('-inv', '--inversion', help='flag of color inverted input image',
                        action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Cell_Segmentation()
