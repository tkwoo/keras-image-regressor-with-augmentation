from __future__ import print_function
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.models import model_from_json
import keras.backend as K
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import models
from utils import centering, un_centering
from utils import get_classmap_keras, get_classmap_numpy
from utils import image_read
from utils import plot_confusion_matrix
import utils

import face_recognition

from six import next

class predictor:
    def __init__(self, flag):
        self.flag = flag
        self.pd_label = pd.read_csv('./label.csv')

    def name2label(self, name):
        y_label = self.pd_label[self.pd_label['Filename']==name]
        y_label = y_label.values[0][1:]
        return y_label

    def user_generation(self, train_generator):
        # cv2.namedWindow('show', 0)
        # cv2.resizeWindow('show',500, 500)
        for total_iter, images in enumerate(train_generator):
            batch_idx = train_generator.batch_index
            batch_size = images.shape[0]

            if train_generator.batch_index != 0:
                batch_idx = batch_idx - 1
            else:
                batch_idx = train_generator.n//train_generator.batch_size
            
            idx_list = train_generator.index_array[batch_idx*train_generator.batch_size:batch_idx*train_generator.batch_size+batch_size]
            # print (batch_idx, idx_list.shape, train_generator.n)

            list_filenames = np.array(train_generator.filenames)[idx_list]
            
            list_filenames = [os.path.basename(path) for path in list_filenames]
            np_labels = np.array([self.name2label(name) for name in list_filenames], dtype=np.float32)

            np_labels /= 5
            
            yield (images, np_labels)

    def cam(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        
        #model = model_from_json(loaded_model_json)
        model = models.vgg_like(self.flag)

        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'weight*')))
        model.load_weights(weight_list[-1])
        print ('[*] model load : %s'%weight_list[-1])
        
        label_list = [os.path.basename(path) for path 
                    in sorted(glob(os.path.join(self.flag.data_path, '*')))]
        dict_name_list = dict()
        for name in label_list:
            dict_name_list[name] = [path for path 
                    in sorted(glob(os.path.join(self.flag.data_path, name, '*')))]
            dict_name_list[name] = dict_name_list[name][:50]
        
        data_list = []
        [data_list.append(image_read(name, color_mode=1, target_size=flag.image_size)) 
                for name in dict_name_list['dog']]
        # print len(data_list)
        np_Inference_data_list = np.array(data_list)[:,0,:,:,:]
        np_original_data_list = np.array(data_list)[:,1,:,:,:].astype(np.uint8)
        # print np_Inference_data_list.shape

        result = model.predict(np_Inference_data_list, self.flag.batch_size)
        # print result.shape
        prediction_labels = np.argmax(result, axis=1)

        classmap = get_classmap_keras(self.flag, model, np_Inference_data_list)

        assert classmap.shape[0] == np_original_data_list.shape[0] == prediction_labels.shape[0]
        
        cv2.namedWindow("show", 0)
        cv2.resizeWindow("show", 500, 500)
        
        for idx in range(classmap.shape[0]):
            predicted_label = prediction_labels[idx]
            print ("[*] %d's label : %s"%(idx, label_list[predicted_label]))
            # print(np_original_data_list.shape)
            img_original = np_original_data_list[idx,:,:,:]
            img_classmap = classmap[idx,:,:,predicted_label]
            color_classmap = cv2.applyColorMap(img_classmap, cv2.COLORMAP_JET)
            img_show = cv2.addWeighted(img_original, 0.8, color_classmap, 0.5, 0.)
            # cv2.imshow("show", img_show)
            # cv2.imwrite("./result/dog_%d.png"%idx, img_show)
            # if cv2.waitKey(1) == 27:
            #     break
        print ("[*] done")

    def evaluate(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name.split('/')[0], 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # model = models.vgg_like(self.flag)
        if os.path.splitext(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name))[-1] == '.h5':
            weight_file_path = os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name)
        else:
            weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
            weight_file_path = weight_list[-1]
        model.load_weights(weight_file_path)
        model.compile(optimizer='Adam', loss='mse', metrics=['mae'])
        print ("[*] model load : %s"%weight_file_path) #weight_list[-1])
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print ("[*] model loading Time: %.3f ms"%t_total)

        test_datagen = ImageDataGenerator(
            preprocessing_function=centering,
            rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'val'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                shuffle=False,
                color_mode='rgb',
                class_mode=None)

        user_test_gen = self.user_generation(test_generator)

        t_start = cv2.getTickCount()
        loss, acc = model.evaluate_generator(user_test_gen, test_generator.n // self.flag.batch_size)
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print ('[*] test loss : %.4f'%loss)
        print ('[*] test acc  : %.4f'%acc)
        print ("[*] evaluation Time: %.3f ms"%t_total)

        ### confusion matrix
        # pred_generator = test_datagen.flow_from_directory(
        #         os.path.join(self.flag.data_path, 'val'),
        #         target_size=(img_size, img_size),
        #         batch_size=1,
        #         shuffle=False,
        #         #color_mode='grayscale',
        #         class_mode='categorical')

        # pred = model.predict_generator(pred_generator, test_generator.n)
        # y_pred = np.argmax(pred, axis=1)
        # y_true = pred_generator.classes
        
        # cnf_matrix = confusion_matrix(y_true, y_pred)
        # np.set_printoptions(precision=2)
        # Plot non-normalized confusion matrix
        # plt.figure()
        # class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
        # # Plot normalized confusion matrix
        # plt.figure()
        # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')

        # plt.show()
    
    def inference_temp(self):
        img_size = (self.flag.image_width, self.flag.image_height)
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        ### initialize
        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # model = models.vgg_like(self.flag)        
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print ("[*] model load: %s"%weight_list[-1])
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print ("[*] model loading Time: %.3f ms"%t_total)

        ### dataset
        # image_name_list = ['./data/g1.png','./data/iu1.jpg', './data/dh3.jpg', './data/tkwoo2.jpg']
        # image_name_list = ['./data/g1.png', 'data/shlee.jpg', './data/g1.png', './data/g2.png', './data/g3.png', './data/g4.png']
        # image_name_list = ['./data/temp0.png', './data/temp1.png', './data/temp2.png', './data/temp3.png']
        # image_name_list = ['./data/iusuji.jpg', './data/sejung.jpg', './data/sj3.jpg']
        image_name_list = ['./data/yui2.png', './data/yui3.png', './data/yui4.png', './data/yui5.png']
        # gap = 40
        for img_name in image_name_list:
            img, show = image_read(img_name, 1, target_size=None)
            rgb_ori = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            face_location = face_recognition.face_locations(rgb_ori,0,'cnn')
            # face_location = (np.array(face_location) - common_margin).tolist()
            t,r,b,l = face_location[0]
            
            for i in range(1):
                gap_y = int(round((r-l)*0.5))
                gap_x = int(round(((r-l+2*gap_y)*4/3-(r-l))/2))
                t,r,b,l = (t-gap_y,r+gap_x,b+gap_y,l-gap_x)
                print ((b-t)/(r-l))
                t = max([t,0])
                r = min([r,img.shape[1]])
                b = min([b,img.shape[0]])
                l = max([l,0])
                crop_show = show[t:b,l:r]
                crop_img = img[t:b,l:r]
                crop_img = cv2.resize(crop_img, (160,120))
                img_input = np.expand_dims(crop_img, 0)
                
                t_start = cv2.getTickCount()
                result = model.predict(img_input, 1)
                t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
                print ("[*] Processing Time: %.3f ms"%t_total)
                print (os.path.basename(img_name), result[0]*5)
            
            cv2.rectangle(show, (l,t), (r,b), (0,0,0), 2)
            cv2.imshow('show', show)
            key = cv2.waitKey()
            if key == 27:
                exit()
            continue
            
            # cv2.imshow("show", show)
            # if cv2.waitKey(0) == 27:
            #     break
        # data_test.to_csv('./result.csv')
    
    def inference(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # model = models.vgg_like(self.flag)        
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print ("[*] model load: %s"%weight_list[-1])
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print ("[*] model loading Time: %.3f ms"%t_total)

        if os.path.isdir(self.flag.test_image_path):
            image_name_list = utils.get_image_name_list(self.flag.test_image_path)
            # print os.path.splitext(self.flag.test_image_path)
        elif os.path.splitext(self.flag.test_image_path)[-1] == '.png':
            image_name_list = [self.flag.test_image_path]
        elif os.path.splitext(self.flag.test_image_path)[-1] == '.jpg':
            image_name_list = [self.flag.test_image_path]
        # print "[*] Inference data path:", self.flag.test_image_path
        # print "[*] # of data:", len(image_name_list)
        # print image_name_list

        # list_column = ['img_name', 'predicted_label', 'softmax']
        # if os.path.exists('./result.csv'):
        #     data_test = pd.read_csv('./result.csv', index_col=0)
        # else:
        #     fCsvOut = open('./result.csv', 'w')
        #     fCsvOut.write('img_name,predicted_label,softmax\n')
        #     fCsvOut.close()
        #     data_test = pd.read_csv('./result.csv', index_col=0)
        # np_data_test = data_test.values
        
        for img_name in image_name_list:
            img, show = image_read(img_name, 1, target_size=img_size)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # img = np.expand_dims(img, 2)
            img = np.expand_dims(img, 0)
            
            t_start = cv2.getTickCount()
            result = model.predict(img, 1)
            t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
            print ("[*] Processing Time: %.3f ms"%t_total)
            
            print (os.path.basename(img_name),)
            # print result
            
            predict_label = np.argmax(result)
            print (predict_label, )
            # print predict_label.dtype
            # print predict_label.astype(np.str)
            # exit()
            sm_str = ''.join('%0.3f '%e for e in result[0])[:-1]
            sm_stdprint = ''.join('%0.3f,'%e for e in result[0])[:-1]
            print (sm_stdprint)
            
            # inserting_data = [predict_label.astype(np.str), sm_str]
            # data_test.ix[os.path.basename(img_name)] = inserting_data
            
            # cv2.imshow("show", show)
            # if cv2.waitKey(0) == 27:
            #     break
        # data_test.to_csv('./result.csv')

    def inference_MIT(self):
        img_size = (self.flag.image_width, self.flag.image_height)
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        ### initialize
        t_start = cv2.getTickCount()
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        # model = models.vgg_like(self.flag)        
        weight_list = sorted(glob(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, "weight*")))
        model.load_weights(weight_list[-1])
        print ("[*] model load: %s"%weight_list[-1])
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
        print ("[*] model loading Time: %.3f ms"%t_total)

        ### dataset
        image_name_list = sorted(glob('/run/media/tkwoo/myWorkspace/workspace/01.dataset/04.facedataset/MIT_LK_SELF_images/*.png'))
        
        # fOut = open('./MIT_LK_SELF_pred.csv', 'w')
        pd_data = pd.read_csv('./temp.csv', names=['name', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        np_data = pd_data.values

        enter_idx = 0
        for idx, img_name in enumerate(image_name_list):
            if idx%10 == 0 and idx != 0:
                enter_idx += 1
            # if idx < 4080:
                # continue
            img, show = image_read(img_name, 1, target_size=None)
            rgb_ori = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            face_location = face_recognition.face_locations(rgb_ori, 0, 'cnn')
            # face_location = (np.array(face_location) - common_margin).tolist()
            if face_location is None or len(face_location) == 0:
                continue
            t,r,b,l = face_location[0]
            
            gap = 0
            t,r,b,l = (t-gap,r+gap,b+gap,l-gap)
            t = max([t,0])
            r = min([r,img.shape[1]])
            b = min([b,img.shape[0]])
            l = max([l,0])
            crop_show = show[t:b,l:r]
            crop_img = img[t:b,l:r]
            crop_img = cv2.resize(crop_img, (96,96))
            img_input = np.expand_dims(crop_img, 0)
            
            t_start = cv2.getTickCount()
            result = model.predict(img_input, 1)
            t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency() * 1000 
            # print ("[*] Processing Time: %.3f ms"%t_total)
            print (idx, os.path.basename(img_name), result[0]*5)
            # print (type('%.3f'%float(result[0][0]*5)))
            
            parsed_name = os.path.basename(img_name).split('_')
            human_id = parsed_name[1]
            human_idx = parsed_name[2].split('.')[0]
            # print (human_idx)
            # print (type(np_data[idx][0]), type(human_id))
            if int(np_data[enter_idx][0]) != int(human_id):
                print ('wrong')
                print (int(np_data[idx][0]), int(human_id))
                exit()
            np_data[enter_idx][int(human_idx)+1] = '%.3f'%float(result[0][0]*5)

            pd_out = pd.DataFrame(data=np_data, columns=['name', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            pd_out.to_csv('result_%02d.csv'%gap)

            cv2.rectangle(show, (l,t), (r,b), (0,0,0), 2)
            cv2.imshow('show', show)
            key = cv2.waitKey(1)
            if key == 27:
                exit()
            continue

        