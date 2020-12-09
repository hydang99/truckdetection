import cv2
import numpy as np
import os


mypath = 'ORIGINAL_IMAGES/'
(_, _, filenames) = next(os.walk(mypath))
newPath = 'AUGMENTED_IMAGES/'


for f in filenames:
    path = mypath + str(f)
    fpath = newPath + str(f)
    img = cv2.imread(path)
    img[img != 0] = 255 # change everything to white where pixel is not black
    cv2.imwrite(fpath, img)