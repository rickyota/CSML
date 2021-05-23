from datetime import datetime
import argparse
import textwrap
import time

from . import train
from . import infer


def Cell_Segmentation():

    print("Start Cell Segmentation.")
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

    args = argument()

    if not args.finfer:
        print("Train and infer.")
        try:
            start = time.time()

            train.train_step(fname_train=args.train,
                             fname_label=args.label,
                             fname_model=args.model,
                             N_test=args.ntest, N_train=args.ntrain, N_epoch=args.nepoch, batchsize=args.nbatch,
                             hgh=args.height, wid=args.width,
                             mode=args.mode)
            # train.train_step(fname_train=os.path.join("data", args.train),
            #                 fname_label=os.path.join("data", args.label),
            #                 fname_model=os.path.join("data", args.model),
            #                 N_test=args.ntest, N_train=args.ntrain, N_epoch=args.nepoch, batchsize=args.nbatch,
            #                 hgh=args.height, wid=args.width,
            #                 mode=args.mode)
            print("train time", time.time() - start)
        except Exception as e:
            raise Exception(e, "Got an error in training step.")
        try:
            start = time.time()
            infer.infer_step(fname_infer=args.infer,
                             fname_save=args.output,
                             fname_model=args.model,
                             thre_discard=args.discard, wid_dilate=args.close,
                             fstats=args.nostats)
            # infer.infer_step(fname_infer=os.path.join("data", args.infer),
            #                 fname_save=os.path.join("result", args.output),
            #                 fname_model=os.path.join("data", args.model),
            #                 thre_discard=args.discard, wid_dilate=args.close,
            #                 fstats=args.nostats)
            print("infer time", time.time() - start)
        except Exception as e:
            raise Exception(e, "Got an error in inference step.")

        print("All done.")

    else:
        print("Only Infer.")
        try:
            start = time.time()
            infer.infer_step(fname_infer=args.infer,
                             fname_save=args.output,
                             fname_model=args.model,
                             thre_discard=args.discard, wid_dilate=args.close,
                             fstats=args.nostats)
            # infer.infer_step(fname_infer=os.path.join("data", args.infer),
            #                 fname_save=os.path.join("result", args.output),
            #                 fname_model=os.path.join("data", args.model),
            #                 thre_discard=args.discard, wid_dilate=args.close,
            #                 fstats=args.nostats)
            print("infer time", time.time() - start)
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
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-f', '--finfer',
                        help='Only inferrence',
                        action='store_true')
    parser.add_argument('-t', '--train', help='Train folder name',
                        default='train')
    parser.add_argument('-l', '--label', help='Label folder name',
                        default='label')
    parser.add_argument('-i', '--infer', help='Infer folder name',
                        default='infer')
    parser.add_argument('-o', '--output', help='Output folder name',
                        default='result')
    parser.add_argument('-m', '--model', help='Model file name',
                        default='model.pkl')

    parser.add_argument('-nr', '--ntrain', help='Number of training images',
                        type=int, default=20000)
    parser.add_argument('-ns', '--ntest', help='Number of testing images',
                        type=int, default=3000)
    parser.add_argument('-he', '--height', help='Height of patches',
                        type=int, default=64)
    parser.add_argument('-wi', '--width', help='Width of patches',
                        type=int, default=64)
    parser.add_argument('-td', '--discard', help='Threshold of discarding',
                        type=int, default=100)
    parser.add_argument('-wc', '--close', help='Width of closing',
                        type=int, default=1)
    parser.add_argument('-ne', '--nepoch', help='Number of epoch',
                        type=int, default=1)
    parser.add_argument('-nb', '--nbatch', help='Number of batch',
                        type=int, default=100)
    parser.add_argument('-mo', '--mode',
                        help=textwrap.dedent('''\
                        Way to choose train patches
                        back(default): the center of patch is not on background
                        whole: whole patch does not overlap with background, use this option when only part of image is labelled.
                        all: all
                        '''),
                        default='back')

    parser.add_argument('-s', '--nostats',
                        help='Do not return stats. Much faster.',
                        action='store_false')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Cell_Segmentation()
