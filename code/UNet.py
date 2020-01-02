#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:06:39 2019

@author: yahelsalomon
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

class UNet():
    def __init__(self, model_name,input_shape,num_classes,param_dict):
        self.name = model_name
        self.input_shape = input_shape
        self.output_shape = (0,0,0)
        self.num_classes = num_classes
        self.param_dict = param_dict
        self.acc = 0
        self.pred = 0
        self.model = None

    def conv_block(self,layer_in,mul_num): 
        layer_out = Conv2D(16*mul_num, kernel_size = (3, 3) ,  padding="same")(layer_in)
        layer_out = BatchNormalization()(layer_out)
        layer_out = Activation("relu")(layer_out)
        layer_out = Conv2D(16*mul_num, kernel_size = (3, 3),  padding="same")(layer_out)
        layer_out = BatchNormalization()(layer_out)
        layer_out = Activation("relu")(layer_out)
        return layer_out
    
    def dconv_block(self,layer_in,model_1,mul_num): 
        model_2 = Conv2DTranspose(16*mul_num, kernel_size = (3, 3), strides=(2, 2), padding='same')(layer_in)
        layer_out = Add()([model_2,model_1])
        layer_out = Dropout(self.param_dict['drop'])(layer_out)
        layer_out = Conv2D(16*mul_num,kernel_size = (3, 3 ),  padding="same")(layer_out)
        layer_out = BatchNormalization()(layer_out)
        layer_out = Activation("relu")(layer_out)
        layer_out = Conv2D(16*mul_num, kernel_size = (3, 3),  padding="same")(layer_out)
        layer_out = BatchNormalization()(layer_out)
        layer_out = Activation("relu")(layer_out)
        return layer_out
    
    def create_model(self):
        model_input = Input(self.input_shape, name='model_input')
        
        down_1 = self.conv_block(model_input,1)
        p1 = MaxPooling2D(pool_size=(2,2))(down_1)
        p1 = Dropout(self.param_dict['drop'])(p1)
    
        down_2 = self.conv_block(p1,2)
        p2 = MaxPooling2D(pool_size=(2,2))(down_2)
        p2 = Dropout(self.param_dict['drop'])(p2)
        
        down_3 = self.conv_block(p2,4)
        p3 = MaxPooling2D(pool_size=(2,2))(down_3)
        p3 = Dropout(self.param_dict['drop'])(p3)
    
        down_4 = self.conv_block(p3,8)
        p4 = MaxPooling2D(pool_size=(2,2))(down_4)
        p4 = Dropout(self.param_dict['drop'])(p4)
    
        down_5 = self.conv_block(p4,16)
        
        print(down_5.shape, down_4.shape)
        up_1 = self.dconv_block(down_5,down_4,8)
        up_2 = self.dconv_block(up_1,down_3,4)
        up_3 = self.dconv_block(up_2,down_2,2)
        up_4 = self.dconv_block(up_3,down_1,1)


        
        fully_layer_1 = Dense(32, activation='relu')(up_4)
        fully_layer_2 = Dense(64, activation='relu')(fully_layer_1)
        model_output = Dense(self.num_classes, activation='sigmoid', name='model_output')(fully_layer_2)
        
        model = Model(inputs=[model_input],outputs=[model_output])
        
        self.output_shape = (model_output.shape[1],model_output.shape[2],model_output.shape[3])
        
        self.set_model(model)
        
    def get_model(self):
        return self.model
    def set_model(self, updated):
        self.model = updated
        
    
    
