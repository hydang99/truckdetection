import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import os
from os.path import isfile

mypath = 'ORIGINAL_IMAGES/'
(_, _, filenames) = next(os.walk(mypath))

for f in filenames:
    print(f)
    path = mypath + str(f)
    image = io.imread(path)
    #Rotated Images
    rotated1 = rotate(image, angle = 45, mode = 'wrap')
    rotated2 = rotate(image, angle = 135, mode = 'wrap')

    io.imsave("AUGMENTED_IMAGES/{}_r45.png".format(f), rotated1)
    io.imsave("AUGMENTED_IMAGES/{}_r135.png".format(f), rotated2)
    
    #Flipped Images
    flipR = np.fliplr(image)
    flipU = np.flipud(image)

    io.imsave("AUGMENTED_IMAGES/{}_flipR.png".format(f), flipR)
    io.imsave("AUGMENTED_IMAGES/{}_flipU.png".format(f), flipU)

    #Noisy Images
    s = 0.155
    noisyRandom = random_noise(image,var = s**2)
    io.imsave("AUGMENTED_IMAGES/{}_noisy.png".format(f), noisyRandom)

    #BlurryImages
    blurred = gaussian(image,sigma = 1,multichannel=True)
    io.imsave("AUGMENTED_IMAGES/{}_blurry.png".format(f), blurred)

    

