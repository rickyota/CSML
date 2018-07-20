import os
from datetime import datetime
import argparse

from train import train_step
from infer import infer_step


def Cell_Segmentation():

    print("Start Cell Segmentation.")
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

    args = argument()

    if not args.flaginfer:
        print("Train and infer.")
        try:
            train_step(fname_train=os.path.join("data", args.train),
                       fname_label=os.path.join("data", args.label),
                       fname_model=os.path.join("data", args.model),
                       N_test=args.ntest, N_train=args.ntrain, N_epoch=args.nepoch, batchsize=args.nbatch,
                       hgh=args.height, wid=args.width,
                       mode=args.mode)
        except Exception as e:
            raise Exception(e, "Got an error in training step.")
        try:
            infer_step(fname_infer=os.path.join("data", args.infer),
                       fname_save=os.path.join("result", args.output),
                       fname_model=os.path.join("data", args.model),
                       thre_discard=args.discard, wid_dilate=args.dilate)
        except Exception as e:
            raise Exception(e, "Got an error in inference step.")

        print("All done.")

    else:
        print("Only Infer.")
        try:
            infer_step(fname_infer=os.path.join("data", args.infer),
                       fname_save=os.path.join("result", args.output),
                       fname_model=os.path.join("data", args.model),
                       thre_discard=args.discard, wid_dilate=args.dilate)
        except Exception as e:
            raise Exception(e, "Got an error in inference step.")

        print("All done.")

        print("afteradd")


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
