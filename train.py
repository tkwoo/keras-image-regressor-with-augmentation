from __future__ import print_function
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import set_session
from keras.utils import plot_model
import tensorflow as tf
import keras
import cv2
import numpy as np
import os
from glob import glob
import argparse
import random
import math
import pandas as pd

import models
import callbacks
from utils import centering
from keras.optimizers import Adam

class Trainer:
    def __init__(self, flag):
        self.flag = flag
        self.pd_label = pd.read_csv('./label.csv')

    def name2label(self, name):
        y_label = self.pd_label[self.pd_label['Filename']==name]
        y_label = y_label.values[0][1:]
        return y_label

    def user_generation(self, train_generator):
        for total_iter, images in enumerate(train_generator):
            batch_idx = train_generator.batch_index
            batch_size = images.shape[0] #train_generator.batch_size
            # idx = train_generator.index_array[0 + (batch_idx-1)*batch_size]
            idx_list = train_generator.index_array[batch_idx-1:batch_idx-1+batch_size]
            # print (train_generator.filenames[idx])
            # list_filenames = train_generator.filenames[idx:idx+batch_size]
            print (batch_idx, idx_list.shape)

            list_filenames = np.array(train_generator.filenames)[idx_list]
            
            list_filenames = [os.path.basename(path) for path in list_filenames]
            np_labels = np.array([self.name2label(name) for name in list_filenames], dtype=np.float32)
            np_labels /= 5.1
            if len(images) != len(np_labels):
                print("error")
                print (images.shape, np_labels.shape)
                print(list_filenames)
                print(np_labels)
                continue
            exit()
            # yield (images, np_labels)

    def lr_step_decay(self, epoch):
        init_lr = self.flag.initial_learning_rate
        lr_decay = self.flag.learning_rate_decay_factor
        epoch_per_decay = self.flag.epoch_per_decay
        lrate = init_lr * math.pow(lr_decay, math.floor((1+epoch)/epoch_per_decay))
        # print lrate
        return lrate

    def train(self):
        img_size = self.flag.image_size
        batch_size = self.flag.batch_size
        epochs = self.flag.total_epoch

        train_datagen = ImageDataGenerator(
            preprocessing_function=centering,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
            )

        test_datagen = ImageDataGenerator(
            preprocessing_function=centering,
            rescale=1./255
            )

        train_generator = train_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'train'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                color_mode='rgb',
                class_mode=None,
                shuffle=True
                )

        validation_generator = test_datagen.flow_from_directory(
                os.path.join(self.flag.data_path, 'val'),
                target_size=(img_size, img_size),
                batch_size=batch_size,
                shuffle=False,
                color_mode='rgb',
                class_mode=None)

        user_train_generator = self.user_generation(train_generator)
        user_val_generator = self.user_generation(validation_generator)

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        lr = self.flag.initial_learning_rate
        model = models.vgg_like_regressor(self.flag)
                
        model.compile(optimizer=Adam(lr=lr, decay=1e-6), loss='mse')
        
        if self.flag.pretrained_weight_path != None:
            model.load_weights(self.flag.pretrained_weight_path)
            print ("[*] loaded pretrained model: %s"%self.flag.pretrained_weight_path)
        if not os.path.exists(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name)):
            os.mkdir(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name))
        model_json = model.to_json()
        with open(os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.json'), 'w') as json_file:
            json_file.write(model_json)
        plot_model(model, to_file=os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'model.png'))
        
        log_file_path = os.path.join(self.flag.ckpt_dir, self.flag.ckpt_name, 'training.log')
        csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=7)

        model_checkpoint = ModelCheckpoint(
                    os.path.join(
                                self.flag.ckpt_dir, 
                                self.flag.ckpt_name,
                                'weights.{epoch:02d}.h5'), 
                    save_best_only=False,
                    verbose=1,
                    monitor='val_loss',
                    period= 2, #self.flag.total_epoch // 10 + 1, 
                    save_weights_only=True)
        learning_rate = LearningRateScheduler(self.lr_step_decay)
        # vis = callbacks.trainCheck(self.flag)

        callback_list = [model_checkpoint, learning_rate, csv_logger]#, vis]
        model.fit_generator(
            user_train_generator,
            steps_per_epoch=train_generator.n // batch_size,
            epochs=epochs,
            validation_data=user_val_generator,
            validation_steps=train_generator.n // batch_size,
            callbacks=callback_list
        )