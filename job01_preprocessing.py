from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = './imgs/horse-or-human/**/'
categories = ['horse', 'human']
image_w = 64
image_h = 64
# pixel = 64 * 64 * 3
X = []
Y = []

for idx, category in enumerate(categories):
    for i, img_path in enumerate(glob.glob(img_dir + category + '*.png')):   #모든 경로의 리스트
        try:
            img = Image.open(img_path)
            img = img.resize((image_w, image_h))
            img = img.convert("RGB")
            img = np.array(img)
            X.append(img)
            Y.append(idx)
            if i % 300 == 0:
                print(category, ':', img_path)
        except:
            print('error :', category, img_path)

X = np.array(X)
Y = np.array(Y)
X = X/255   # 스케일링
print(X[0])
print(Y[0])
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
np.save('binary_data/horse_human_x_train.npy', X_train)
np.save('binary_data/horse_human_y_train.npy', Y_train)
np.save('binary_data/horse_human_x_test.npy', X_test)
np.save('binary_data/horse_human_y_test.npy', Y_test)
