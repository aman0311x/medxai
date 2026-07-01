import numpy as np

def crop_roi(image, bbox):
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 elements: [ymin, xmin, ymax, xmax]")

    ymin, xmin, ymax, xmax = map(int, map(np.round, bbox))

    height, width = image.shape[:2]
    ymin = max(0, ymin)
    xmin = max(0, xmin)
    ymax = min(height, ymax)
    xmax = min(width, xmax)

    if ymin >= ymax or xmin >= xmax:
        raise ValueError("Invalid ROI: ymin >= ymax or xmin >= xmax")

    return image[ymin:ymax, xmin:xmax]