from __future__ import print_function
import keras
import cv2
import numpy as np
import os
import utils
from glob import glob
from utils import image_read

class trainCheck(keras.callbacks.Callback):
    def __init__(self, flag):
        self.flag = flag
        self.epoch = 0

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        # self.train_visualization(self.model, epoch)
        return

    def on_epoch_end(self, epoch, logs={}):
        # self.train_visualization(self.model, epoch)
		# self.losses.append(logs.get('loss'))
		# y_pred = self.model.predict(self.model.validation_data[0])
		# self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        if batch%20 == 0:
            self.train_visualization(self.model, batch)
        return
    
    def train_visualization(self, model, index):
        image_name_list = sorted(glob(os.path.join(self.flag.data_path, 'val','01.*','*.png')))
        image_name = image_name_list[-1]
        imgInput, imgShow = image_read(image_name, self.flag.color_mode, self.flag.image_size)
        
        output_path = os.path.join(self.flag.output_dir, self.flag.ckpt_name)
        if os.path.exists(output_path) == False:
            utils.mkdir_p(output_path)
        
        input_data = imgInput.reshape((1,
                        self.flag.image_size,
                        self.flag.image_size,
                        self.flag.color_mode*2+1)) # color: 3, gray: 1
        t_start = cv2.getTickCount()
        result = model.predict(input_data, 1)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000
        print ("\n[*] Predict Time: %.3f ms"%t_total)
        
        prediction_labels = np.argmax(result, axis=1)
        classmap = utils.get_classmap_keras(self.flag, self.model, input_data, prediction_labels[0])

        imgMask = classmap[0,:,:].astype(np.uint8)
        if self.flag.color_mode == 0:
            imgShow = cv2.cvtColor(imgShow, cv2.COLOR_GRAY2BGR)
        else:
            imgShow = imgShow
        # _, imgMask = cv2.threshold(imgMask, int(255*flag.confidence_value), 255, cv2.THRESH_BINARY)
        imgMaskColor = cv2.applyColorMap(imgMask, cv2.COLORMAP_JET)
        # imgZero = np.zeros((256,256), np.uint8)
        # imgMaskColor = cv2.merge((imgZero, imgMask, imgMask))
        imgShow = cv2.addWeighted(imgShow, 0.9, imgMaskColor, 0.5, 0.0)
        
        output_path = os.path.join(output_path, 
                            '%s_%s_'%(self.epoch, index)+os.path.basename(image_name))
        cv2.imwrite(output_path, imgShow)
        print ("[*] SAVE:[%s]"%output_path)
        # cv2.imwrite(os.path.join(output_path, 'img%04d.png'%epoch), imgShow)
        # cv2.namedWindow("show", 0)
        # cv2.resizeWindow("show", 800, 800)
        # cv2.imshow("show", imgShow)
        # cv2.waitKey(1)