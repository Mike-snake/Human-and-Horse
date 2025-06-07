from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = './imgs/horse-or-human/horses'
categories = ['horses', 'humans']
png_files = glob.glob('./imgs/horse-or-human/horses/*.png')

print(png_files)
