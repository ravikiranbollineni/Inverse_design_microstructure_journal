# In[1]:


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import io
import imageio
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
img_collection=[]
ua = []
im = []
img_me_data = []
img_me_arr = []
with open('Input_geometry_2000.pkl', 'rb') as f1:
    img_collection = pickle.load(f1,encoding='latin1')
img_dim =(96,96,1)
lok = img_collection.reshape(((len(img_collection),)+ img_dim))
lok = lok/255.0
######## Image input mean ################
for ch in range(0,2001):
    lok_2 = lok[ch].mean(axis=(0, 1))
    img_me_data.append(lok_2)
img_me_data = np.array(img_me_data)

for k in range (0,2001):
    for l in range(11):
        in_imag = np.full((96, 96), img_me_data[k])
        img_me_arr.append(in_imag) 
img_me_arr = np.array(img_me_arr)
######## Image input array ################
for m in range(0, 5001):
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])
    im.append(lok[m])

ua0 =[]
ua1 =[]
ua2 =[]
with open('Output_Mises_stress_2000.pkl', 'rb') as f1:
    ua0 = pickle.load(f1,encoding='latin1')    
    
    
img_dim =(96,96,1)
img_dim2 =(96,96,2)

sa0 =[]
sa1 =[]
sa2 =[]
with open('Output_stress_2000.pkl', 'rb') as f4:
    sa0 = pickle.load(f4,encoding='latin1')    
    
ra = np.array(im)
ua0 = np.array(ua0)
sa0 = np.array(sa0)

ra = ra.reshape((2001,)+(11,)+ img_dim) ######Input geometry
img_me_arr = img_me_arr.reshape((2001,)+(11,)+ img_dim) ######Input geometry mean 
ua0 = ua0.reshape((2001,)+(11,)+ img_dim) 
ua0 = ua0[:2001]

sa0 = sa0.reshape((2001,)+(11,)+ img_dim)
sa = np.concatenate((sa0), axis=0) ######Output constant stress value each step
ma = np.concatenate((ua0, sa0),axis= -1)######Output both Mises stress and constant stress values
ki = np.concatenate((ra0, img_me_arr),axis= -1)######Output both image data and stress-strain data


ki.shape
ma.shape



####data partetion 90 percent for training and remaining for testing and validation
x_train = ki[:1501,:]
y_train = ma[:1501,:]
x_val = ki[1501:2001,:]
y_val = ma[1501:2001,:]

x_test = ki[1981:2001,:]
x_test = x_test.reshape((20,)+(11,)+ img_dim2)
y_test = ma[1981:2001,:]
y_test = y_test.reshape((20,)+(11,)+ img_dim2)


# In[5]:


y_val.shape



#Images pass through the CONVLST layers
inp = layers.Input(shape=(None, *x_train.shape[2:]))

# We will construct 3 `ConvLSTM2D` layers with batch normalization,
# followed by a `Conv3D` layer for the spatiotemporal outputs.
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(4, 4),
    padding="same",
    return_sequences=True,
    activation="relu",
)(inp)
#x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(4, 4),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
#x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(4, 4),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
#x = layers.BatchNormalization()(x)
x = layers.ConvLSTM2D(
    filters=64,
    kernel_size=(4, 4),
    padding="same",
    return_sequences=True,
    activation="relu",
)(x)
x = layers.Conv2D(
    filters=2, kernel_size=(1, 1), activation="linear", padding="same"
)(x)

# Next, we will build the complete model and compile it.
model = keras.models.Model(inp, x)
model.compile(
    loss=keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.95 , beta_2=0.999,amsgrad=False),
)


# In[9]:


model.summary()


# In[ ]:


# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

# Define modifiable training hyperparameters.
epochs = 15
batch_size =30

# Fit the model to the training data.
history =model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr],)


# In[ ]:


fig2 = plt.gcf()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.figsize"] = (5.118, 3.543)
plt.rcParams["figure.facecolor"] = 'w'
plt.rcParams["figure.edgecolor"] = 'k'
SMALL_SIZE = 12
MEDIUM_SIZE = 18
BIGGER_SIZE = 18

#------------ all above is Reza's settings-------#
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.tick_params(right= True,top= True,  direction='in')

plt.semilogy(history.history['loss'], 'tab:orange', linestyle='dashed', linewidth=2)
plt.semilogy(history.history['val_loss'], 'tab:blue', linestyle='solid',linewidth=2)
plt.title('Loss vs Epoch')  
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train','Val'], frameon=False, loc = 'upper right')


plt.savefig('train_val_loss_v2_6.eps',bbox_inches='tight', format='eps', dpi=1200 )
    
plt.show()


# In[ ]:


model.save('model_convlstm_5000_V21_Conv2D_kernal_1_64_4_filters.h5')

