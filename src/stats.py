from cv2 import connectedComponentsWithStats
import pandas as pd

from matplotlib import pyplot as plt
import numpy as np


def tag(im):

    _, im_label, stats, centroids = connectedComponentsWithStats(
        np.array(255 - im, np.uint8), connectivity=4)

    stats = np.concatenate([centroids, stats], axis=1)

    df = pd.DataFrame(data=stats,
                      columns=['centroid_x', 'centroid_y',
                               'left', 'top', 'width', 'height', 'area'])

    return im_label, df
