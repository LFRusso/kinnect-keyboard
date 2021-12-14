import mahotas as mt
from skimage.feature import hog

def extract_features(img_gray, img_mask):
    zernike = mt.features.zernike_moments(img_gray, 3)
    fd = hog(img_mask, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2))
    return list(zernike)+list(fd)