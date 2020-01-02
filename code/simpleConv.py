# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:16:38 2019

@author: yahelsalomon
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *


class simpleConv():
    def __init__(self, model_name,input_shape,num_classes,param_dict):
        self.name = model_name
        self.input_shape = input_shape
        self.output_shape = (0,0,0)
        self.num_classes = num_classes
        self.param_dict = param_dict
        self.acc = 0
        self.pred = 0
        self.model = None
    
    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size = (4,4), strides = (1,1), padding = 'same', activation='relu', input_shape=self.input_shape, use_bias=True))
        model.add(Conv2D(32*2, kernel_size = (4,4), strides = (1,1), padding = 'same', activation='relu', use_bias=True))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='sigmoid'))
        self.set_model(model)
        
    def get_model(self):
        return self.model
    def set_model(self, updated):
        self.model = updated