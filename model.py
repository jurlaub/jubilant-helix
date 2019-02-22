import csv
from scipy import ndimage
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from math import ceil
import sklearn


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
# from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D, MaxPooling2D


class Driver(object):
    def __init__(self, fn='model15.h5'):
        self.lines = []
        self.images = []
        self.measurements = []
        self.model = None
        self.filename = fn
              
        self.X_train = None  # set in pre_collect_data
        self.y_train = None  # set in pre_collect_data
        self.validation = None
        
        # --- generator ---
        self.gen_batch_size = 32
        self.image_count = 0
        self.samples_train = 0
        self.samles_valid = 0
        
        self.ceil_crop = 60
        self.floor_crop = 140
        self.train_generator = None
        self.valid_generator = None
        
    def pre_open_csv(self, logName):
        print("pre_open_csv - starting")
        with open(logName) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.lines.append(line)  
                
        print("pre_open_csv - end")
    
    def pre_collect_data(self):
        """ partially from lesson behavior cloning #8
            
            imports center image and measurement & flips image and measurement for data augmentation
        
        """
        print("pre_collect_data - starting")
        # --- collect images and data ---
        for line in self.lines:
            center_image_name = line[0]
            image = ndimage.imread(center_image_name)
            measure = float(line[3])
            
            # -- normal --
            self.images.append(image)
            self.measurements.append(measure)
            # -- invert for augmentation --
            self.images.append(cv2.flip(image, 1))
            self.measurements.append(measure*-1.0)
            
        print("pre_collect_data - np conversion")
        # --- convert to np arrays ---
        self.measurements = np.array(self.measurements)
        self.images = np.array(self.images)
        self.X_train = self.images
        self.y_train = self.measurements

    def _generator(self, samples):
        num_samples = len(samples)
        while 1:
            print("num samples:{}".format(num_samples))
            shuffle(samples)
            for offset in range(0, num_samples, self.gen_batch_size):
                tmp_r = num_samples - offset
                if(tmp_r < self.gen_batch_size):
                    batch_samples = samples[offset:]
                else:
                    batch_samples = samples[offset:offset+self.gen_batch_size]
                
                images = []
                angles = []
                for batch_sample in batch_samples:
                    
                    fn = batch_sample[0] # alter file path here
                    image = ndimage.imread(fn)
                    measure = float(batch_sample[3])
                    images.append(image)
                    angles.append(measure)
                
                # trimming 
                X_train = np.array(images)
                #                 X_train = X_train[self.ceil_crop:self.floor_crop, :,] # crop off the upper and lower portion of the image
                y_train = np.array(angles)
#                 print("shape x_train:{} y_train:{}".format(X_train.shape, y_train.shape))
                yield sklearn.utils.shuffle(X_train, y_train)
               
        
    def pre_collect_data_gen_help(self):
        """ Adapted from project behavioral cloning lesson 18 """
        
        self.samples_train, self.samples_valid = train_test_split(self.lines, test_size=0.2)
        #         print("train:{} valid:{}".format(len(self.samples_train), len(self.samples_valid)))
        
        #         print("valid/batch{}, ceil:{}".format(len(self.samples_valid)/self.gen_batch_size, ceil(len(self.samples_valid)/self.gen_batch_size)))
        self.train_generator = self._generator(self.samples_train)
        self.valid_generator = self._generator(self.samples_valid)
        print("end of precollected data")    
            
            
    def compile_generator_model(self):
        """ MUST call a method that assembles a model before calling this compile method"""
        self.model.compile(loss='mse', optimizer='adam')
        print("compiled")
        self.model.fit_generator( self.train_generator, steps_per_epoch=ceil(len(self.samples_train)/self.gen_batch_size),
                                validation_data=self.valid_generator, validation_steps=ceil(len(self.samples_valid)/self.gen_batch_size), epochs=12,verbose=2)
        #             validation_steps=len(self.samples_valid)                    validation_steps=ceil(len(self.samples_valid)/self.gen_batch_size), epochs=3,verbose=2)
        # --- save data---
        self.model.save(self.filename) 
        
        
    def basic_model(self, generator=False):
        self.model = Sequential()      
        # --- layers ----
        self.model.add(Flatten())
        self.model.add(Dense(1))
        
        if(not generator):
            self.model.compile(loss='mse', optimizer='adam')
            self.model.fit(self.X_train, self.y_train, validation_split=0.2, shuffle=True)
            self.model.save('model1.h5')
        
    def model_lenet_like(self, generator=False):
        
        self.model = Sequential()
        
        if(not generator):
            ishape=(160,320,3)
            TopCrop = 50
            BottomCrop = 20
            # --- preprocess ----
            self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=ishape ))
            self.model.add(Cropping2D(cropping=((TopCrop,BottomCrop), (0,0))))        
        else:
            #             ishape=(80,320,3) # 
            ishape=(160,320,3)
            # --- preprocess ----
            self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=ishape ))
         
        # -- layer 1 --
        self.model.add(Conv2D(6, kernel_size=(5,5),
                 activation='relu', padding='valid'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        # -- layer 2 --
        self.model.add(Conv2D(16, kernel_size=(3,3),
                 activation='relu', padding='valid'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        # -- layer 3 --
        self.model.add(Flatten())
        
        self.model.add(Dense(120))
        
        self.model.add(Dense(84))
        
#         self.model.add(Dropout(.20))
        self.model.add(Dense(1))
        if(not generator):
            self.model.compile(loss='mse', optimizer='adam')
            self.model.fit(self.X_train, self.y_train, 
                       validation_split=0.2, shuffle=True, epochs=5)
            # --- save data---
            self.model.save('model2.h5')  

    def nvidia_like(self):
        """ from lesson slide 15
        """
        ishape=(160,320,3)
        TopCrop = 70
        BottomCrop = 25
        self.model = Sequential()
        # --- preprocess ----
        
        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=ishape ))
        self.model.add(Cropping2D(cropping=((TopCrop,BottomCrop), (0,0))))
        self.model.add(Dropout(0.40))
        
        #         self.model.add(Conv2D(3, kernel_size=(5,5), activation="relu"))   
        self.model.add(Conv2D(24, kernel_size=(2,2), strides=(2,2), activation="relu"))
        self.model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
        self.model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
        # self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) 
        self.model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) 
        self.model.add(Flatten())
        self.model.add(Dropout(0.4))
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))

        
if __name__=="__main__":
    fn = 'model16.h5'
    driver = Driver(fn)
    logname = "driving_log.csv"
    folder1 = 'train1/'
    folder2 = 'train2/brown_c1'
    folder3 = 'train2/brown_c2'
    folder4 = 'train2/bridge_c1'
    
    folder5 = 'train2/full2'
    folder6 = 'train2/f_error'
    folder7 = 'train2/brown_turn3'
    folder8 = 'train2/brown_turn4'
    folder9 = 'train2/bturn5'
    folder10 = 'train2/bturn6'
    folder11 = 'train2/bturn7'
    folder12 = 'train2/bturn8'
    folder13 = 'train2/bturn9'
    folder14 = 'train2/bturn10'
    fn = "{}/{}".format(folder1, logname)
    driver.pre_open_csv(fn)
    fn = "{}/{}".format(folder2, logname)
    driver.pre_open_csv(fn)
    fn = "{}/{}".format(folder3, logname)
    driver.pre_open_csv(fn)
    fn = "{}/{}".format(folder4, logname)
    driver.pre_open_csv(fn)
    
    fn = "{}/{}".format(folder5, logname)
    driver.pre_open_csv(fn)
    fn = "{}/{}".format(folder6, logname)
    driver.pre_open_csv(fn)    
 
    fn = "{}/{}".format(folder7, logname)
    driver.pre_open_csv(fn) 
    
    fn = "{}/{}".format(folder8, logname)
    driver.pre_open_csv(fn)
    fn = "{}/{}".format(folder9, logname)
    driver.pre_open_csv(fn)
    
    fn = "{}/{}".format(folder10, logname)
    driver.pre_open_csv(fn)
    
    fn = "{}/{}".format(folder11, logname)
    driver.pre_open_csv(fn) 
    
    fn = "{}/{}".format(folder12, logname)
    driver.pre_open_csv(fn)
    fn = "{}/{}".format(folder13, logname)
    driver.pre_open_csv(fn) 
    fn = "{}/{}".format(folder14, logname)
    driver.pre_open_csv(fn) 
    
    #driver.pre_collect_data()
    driver.pre_collect_data_gen_help()
    #     driver.basic_model()
    driver.nvidia_like()    
    
    #     driver.model_lenet_like(True)
    driver.compile_generator_model()

    
    pass

    
    