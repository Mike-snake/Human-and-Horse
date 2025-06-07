from PIL import Image
import numpy as np
from keras.models import load_model

model = load_model('./models/horse_human_mode_0.995.h5')
categories = ['horse', 'human']

img = Image.open('./imgs/img3.jpg')
img = img.resize((64,64))
img = np.array(img)
img = img/255
img = img.reshape(1,64,64,3)

pred = model.predict(img)
print(categories[int(np.around(pred))])




