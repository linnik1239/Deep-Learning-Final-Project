
# coding: utf-8
from IPython import get_ipython
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import numpy as np
import keras
from SegMeasure import *
from functions import *
from keras.models import load_model
from IPython.display import clear_output
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import functools
import keras.metrics
print(tf.__version__)
print(keras.__version__)


#root_dir = '/content/drive/My Drive/Colab Notebooks/Data' 
root_dir = "/home/u24913/Tammy"#'/Users/yahelsalomon/Documents/tammy'#
os.listdir(root_dir)

# In[7]: load hyperparameters
params = [x for x in sys.argv[1:]]
param_dict = {'opt':params[0],
              'lr':float(params[1]),
              'drop':float(params[2]),
              'batch':int(params[3]),
              'augment':int(params[4]),
              'div':int(params[5]),
              'is_blank':int(params[6]),
              'name':params[7]
              }
model_num = int(params[8])

# In[8]:
train_csv = os.path.join(root_dir,'my_train10_1.csv')# 'train.csv')#
val_csv = os.path.join(root_dir, 'val.csv')
train_dataset = create_dataset(train_csv,root_dir)
val_dataset = create_dataset(val_csv,root_dir)
train_dataset = cutImages(train_dataset,param_dict['div'])
val_dataset = cutImages(val_dataset,param_dict['div'])
train_dataset = removeBlankImages(train_dataset,param_dict['is_blank']) #optional
train_dataset=train_dataset.repeat(1) #None # Repeat dataset indefinetly 
val_dataset=val_dataset.repeat(1) #None # Repeat dataset indefinetly 
print("val_dataset",val_dataset)

# In[10]:


import numpy
i=0
for image, seg in train_dataset:
    i+=1
    #print(type(image))
    I = image.numpy()
    S = seg.numpy()
    print(numpy.unique(S))
    plt.figure()
    plt.imshow(I.squeeze())
    plt.show()
    plt.figure()
    plt.imshow(S.squeeze())
    plt.show()
    #if i==10:
    break
plt.close('all')  

# In[]:
# ## Batch the dataset
batch_size = param_dict['augment']*param_dict['div']*param_dict['div']
train_data = train_dataset.batch(batch_size) 
val_data = val_dataset.batch(batch_size) 

# In[12]:

for x_train, y_train in train_data:
    print('x_train Shape: {}, y_train Shape: {}'.format(x_train.shape, 
                                                         y_train.shape))
    break

for x_val, y_val in val_data:
    print('x_test Shape: {}, y_test Shape: {}'.format(x_val.shape, 
                                                         y_val.shape))
    break


# In[14]:

## Keras works with floats, so we must cast the numbers to floats

x_train = tf.cast(x_train, tf.float32)
x_val = tf.cast(x_val, tf.float32)

num_classes = 3


# In[16]:

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
print("y_val_categorical",y_val_categorical.shape)
# In[19]:
input_shape = x_train.shape[1:]
args = [param_dict['name'],input_shape, num_classes,param_dict]
my_model = str2Class(param_dict['name'],args)#UNet('full_UNet',input_shape, num_classes,param_dict)
my_model.create_model()


# In[19]:
		
#saved_model_name = root_dir+"/trained_models/"+"model_"+my_model.name+'_' + param_dict['opt'] + '_lr:' + str(param_dict['lr']) + '_drop:' + str(param_dict['drop']) + '_batch:' + str(param_dict['batch']) +".h5"
print(param_dict)

w_array = np.ones(my_model.output_shape)
custom_loss = functools.partial(weighted_cross_entropy,weights = w_array)
custom_loss.__name__ ='weighted_cross_entropy' 

model_arch = my_model.get_model()
#weights = np.ones((3,))
weights = np.array((0.001,1,1))
print(weights)


model_arch.compile(
    optimizer=param_dict['opt'],
    loss= weighted_categorical_crossentropy(weights), #{'model_output':weighted_cross_entropy},#'categorical_crossentropy',#'binary_crossentropy',#lossFunction(H,W,num_classes,output,x_val)#'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#callbacks = [
    #EarlyStopping(monitor='val_loss', patience=10, verbose=1), #patience: number of epochs with no improvement after which training will be stopped.
    #ReduceLROnPlateau(factor=0.1, patience=5, cooldown=5, min_lr=0.00001, verbose=1), #Reduce learning rate when a metric has stopped improving.
    #ModelCheckpoint(saved_model_name, verbose=1, save_best_only=True, save_weights_only=False) #Save the model after every epoch.
#]


history_data = model_arch.fit(x_train, y_train,
                              epochs=50,
                              #callbacks=callbacks,
                              validation_data=(x_val, y_val_categorical),
                              batch_size = param_dict['batch'],
                              verbose = 2,
                              shuffle=True) 

# plot validation for early stopping
plotHistory(history_data,model_num,root_dir)

#_, my_model.acc = model_arch.evaluate(x_val, y_val)
my_model.pred = model_arch.predict(x_val)
#my_model.pred = model_arch(x_val)

full_im_pred = assemblingParts(my_model.pred,num_classes,param_dict['div'])
my_model.pred = np.array(full_im_pred)

full_y_val = np.array(assemblingParts(y_val,1,param_dict['div']))
  
plotResults(my_model.pred,full_y_val,model_num,root_dir)        
model_arch.save(my_model.name)

# The following two lines should be in your code:
# Define the measure object:
seg_measure = SegMeasure()
# THE NEXT LINE CALLS THE CALCULATION

measure_value = seg_measure(full_y_val, my_model.pred).numpy()
print("measure_value =",measure_value)


