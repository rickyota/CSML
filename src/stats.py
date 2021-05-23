from cv2 import connectedComponents
import pandas as pd
import numpy as np


def tag(im, fstats):

    _, im_label = connectedComponents(
        np.array(255 - im, np.uint8), connectivity=4)

    if fstats:
        from skimage.measure import regionprops
        props = regionprops(im_label, cache=True,
                            coordinates='rc')

        columns = ['centroid_x', 'centroid_y', 'area',
                   'perimeter', 'eccentricity', 'major_axis', 'minor_axis',
                   'orientation', 'solidity']

        stats = []
        for p in props:
            tmp = [p.centroid[1], p.centroid[0], p.area, p.perimeter,
                   p.eccentricity, p.major_axis_length, p.minor_axis_length,
                   p.orientation, p.solidity]
            stats.append(tmp)

        df = pd.DataFrame(data=stats,
                          index=np.arange(1, len(props) + 1),
                          columns=columns)

        return im_label, df
    else:
        return im_label, None
