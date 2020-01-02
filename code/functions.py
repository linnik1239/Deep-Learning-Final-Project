
from IPython import get_ipython
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
import keras.metrics
import numpy as np
from scipy.spatial import distance_matrix
from IPython.display import clear_output
import numpy
from UNet import *
from smallUNet import *
from smallUNet_less import *
from simpleConv import *

# In[1]:

def create_dataset(csv_filename,root_dir):
    record_defaults=[tf.string, tf.string]
    image_dataset = tf.data.experimental.CsvDataset([csv_filename],record_defaults, select_cols = [0,1], header = False)
    path_raw = [os.path.abspath(item[0].numpy()) for item in image_dataset]
    path_seg = [os.path.abspath(item[1].numpy()) for item in image_dataset]
    all_image_paths_raw = [tf.image.decode_image(tf.io.read_file(os.path.join(root_dir,item.decode("utf-8")[0:])),dtype=tf.dtypes.uint16) for item in path_raw]
    all_image_paths_seg = [tf.image.decode_image(tf.io.read_file(os.path.join(root_dir,item.decode("utf-8")[0:]))) for item in path_seg]    
    dataset = tf.data.Dataset.from_tensor_slices((all_image_paths_raw,all_image_paths_seg))
    #dataset = dataset.shuffle(len(all_image_paths_raw))
    return dataset

# In[];
def cutImages(dataset,p):
    for im,_ in dataset: 
        H = im.shape[0]
        W = im.shape[1]
        break
    h = int(H/p)
    w = int(W/p)
    image_array=[]
    seg_array=[]
    for image_batch, seg_batch in dataset:
        I = image_batch.numpy()
        S = seg_batch.numpy()
        i=0
        j=0
        for i in range(p):
            for j in range(p):
                new_image=I[j*h:(j+1)*h,w*i:w*(i+1)]
                image_array.append(tf.convert_to_tensor(new_image))
                new_seg=S[j*h:(j+1)*h,w*i:w*(i+1)]
                seg_array.append(tf.convert_to_tensor(new_seg))
        '''i=0
        j=0
        for i in range(p-1):
            for j in range(p-1):
                new_image=I[int((j+0.5)*h):int((j+1.5)*h),int(w*(i+0.5)):int(w*(i+1.5))]
                image_array.append(tf.convert_to_tensor(new_image))
                new_seg=S[int((j+0.5)*h):int((j+1.5)*h),int(w*(i+0.5)):int(w*(i+1.5))]
                seg_array.append(tf.convert_to_tensor(new_seg))'''
    print("len(image_array) = ",len(image_array))
    print("len(seg_array) = ",len(seg_array))
    new_dataset = tf.data.Dataset.from_tensor_slices((image_array,seg_array))
    return new_dataset

# In[]:
def removeBlankImages(dataset,is_blank):
    if is_blank:
        return dataset
    image_array=[]
    seg_array=[]
    for image_batch, seg_batch in dataset:
        S = seg_batch.numpy()
        if (sum(sum(S))!=0):
            image_array.append(image_batch)
            seg_array.append(seg_batch)
        else:
            continue    
    new_dataset = tf.data.Dataset.from_tensor_slices((image_array,seg_array))
    return new_dataset
# In[2]:
def str2Class(str,args):
    return getattr(sys.modules[__name__], str)(*args)


# In[17]:
def weighted_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
    def loss(y_true,y_pred):
        y_pred /= tf.keras.backend.sum(y_pred,axis=-1,keepdims=True)
        y_pred = tf.keras.backend.clip(y_pred,tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.keras.backend.log(y_pred)*weights 
        loss = - tf.keras.backend.sum(loss,-1)
        return loss
    return loss

# In[18]:

def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return tf.keras.backend.log(y_pred / (1 - y_pred))

def lossF(y_true, y_pred,beta):
    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=beta)
    return tf.reduce_mean(loss)

def weighted_cross_entropy(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    loss = lossF(y_true, y_pred,1)
    return loss
    
# In[]:
def assemblingParts(y_pred,num_classes,p):
    print("y_pred.shape",y_pred.shape)
    h = y_pred.shape[1]
    w = y_pred.shape[2]
    H = int(h*p)
    W = int(w*p)
    full_im_pred = []
    S=np.zeros((H,W,num_classes)) 
    i=0
    j=0
    for seg in y_pred:
        S[j*h:(j+1)*h,w*i:w*(i+1),0:num_classes]=seg
        j+=1
        if j==p:
            j=0
            i+=1
        if i==p:
            i=0
            print("S=",S.shape)
            full_im_pred.append(S.copy())
    return full_im_pred
            
# In[21]:

def plotResults(y_pred,y_val,model_num,root_dir):  
    print("model number: ",model_num) 
    y_pred = numpy.clip(y_pred, 0, 2)
    y_pred = numpy.round(y_pred)
    y_pred = y_pred.astype(int)
    #y_pred = np.round(y_pred)
    y_pred_t_new  = [numpy.argmax(y, axis=2, out=None) for y in y_pred]
    for pr,gt in zip(y_pred_t_new,y_val):
        #gt = gt.numpy()
        gt = gt.squeeze().astype(np.uint8)
        pr = pr.squeeze().astype(np.uint8)
        plt.figure()
        plt.imshow(gt)
        plt.figure()
        plt.imshow(pr)
        #plt.savefig(root_dir+"/images_out/" + str(model_num) + "_pred")
        plt.show()
        matplotlib.image.imsave( root_dir+"/images_out/" + str(model_num) + "_gt" + ".png", gt)
        matplotlib.image.imsave( root_dir+"/images_out/" + str(model_num) + "_pred" + ".png", pr)
        #break
    
def plotHistory(hist_data,model_num,root_dir):
    # list all data in history
    #print(hist_data.history.keys())
    # summarize history for accuracy
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.plot(hist_data.history['accuracy'])
    plt.plot(hist_data.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(122)
    plt.plot(hist_data.history['loss'])
    plt.plot(hist_data.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(root_dir+"/images_out/" + str(model_num) + "_history" + ".png")

