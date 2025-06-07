import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PIL import Image
import numpy as np
from keras.models import load_model

form_class = uic.loadUiType("./human_and_horse_app.ui")[0]

class ExampleApp(QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = ('./imgs/img7.jpg','')

        self.btn_open.clicked.connect(self.btn_clicked_slot)
        self.model = load_model('./models/horse_human_mode_0.995.h5')

    def btn_clicked_slot(self):
        old_path = self.path
        self.path = QFileDialog.getOpenFileName(
            self, "Open File", "./imgs/",
            "Image Files (*.jpg *.png *.jpeg *.bmp *.tiff *.gif);;All Files (*.*)"
        )

        print(self.path)
        if self.path[0] == '':
            self.path = old_path

        try:
            pixmap = QPixmap(self.path[0])
            self.lbl_img.setPixmap(pixmap)
            img = pixmap.toImage()


            img = Image.open(self.path[0])
            img = img.convert('RGB')
            img = img.resize((64,64))

            img = np.array(img)
            img = img/255
            img = img.reshape(1,64,64,3)

            pred = self.model.predict(img)
            print(pred)
            if pred[0][0] > 0.5:
                self.lbl_result.setText('사람 입니다.')

            else:
                self.lbl_result.setText('말 입니다.')

        except:
            print('error')




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ExampleApp()
    main_window.show()
    sys.exit(app.exec_())










