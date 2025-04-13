import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf

#NN model path
def r2_metric(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_pred)))
    r2 = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    return r2

#p11
p11_model = r'...\IITMall_IGOM2p11all14_layers_lr_decay_F.h5'
model_p11 = keras.models.load_model(p11_model,custom_objects={'r2_metric':r2_metric})
#dp12
dp12_model = r'...\IITMall_IGOM2dp12all14_layers_lr_decayv1.h5'
model_dp12 = keras.models.load_model(dp12_model,custom_objects={'r2_metric':r2_metric})
#dp22
dp22_model = r'...\IITMall_IGOM2dp22all14_layers_lr_decay_F.h5'
model_dp22 = keras.models.load_model(dp22_model,custom_objects={'r2_metric':r2_metric})
#dp33
dp33_model = r'...\IITMall_IGOM2dp33all14_layers_lr_decay_F.h5'
model_dp33 = keras.models.load_model(dp33_model,custom_objects={'r2_metric':r2_metric})
#dp44
dp44_model = r'...\IITMall_IGOM2dp44all14_layers_lr_decay_F.h5'
model_dp44 = keras.models.load_model(dp44_model,custom_objects={'r2_metric':r2_metric})
#dp43
dp43_model = r'...\IITMall_IGOM2dp43all14_layers_lr_decayv1.h5'
model_dp43 = keras.models.load_model(dp43_model,custom_objects={'r2_metric':r2_metric})

def log_10(x):
    return np.log10(x)


#Single prediction example
# Input parameters
mr = 1.50
mi = 0.005
asp = 0.5
n = 2.2
theta = 135    #0~180
xsize = 60

xsize_in = log_10(xsize)
mi_in = log_10(mi)
model_in = [[mr, mi_in, asp, n, xsize_in, theta]]
                        
model_in_all = np.array(model_in, dtype='float32')

# Prediction
p11_log = model_p11.predict(model_in_all).flatten() 
dp12 = model_dp12.predict(model_in_all).flatten()
dp22 = model_dp22.predict(model_in_all).flatten()
dp33 = model_dp33.predict(model_in_all).flatten()
dp44 = model_dp44.predict(model_in_all).flatten()
dp43 = model_dp43.predict(model_in_all).flatten()




#Batch prediction example
# Input parameters
mrs = [1.50]
mis = [0.005,0.00001]
asps = [0.5,0.6]
ns = [1.4,1.6]
thetas = np.arange(0, 181, 1)
xsizes = [55,60,80,85,100,110,120,180,190,200]


for mr in mrs:
    for mi in mis:
        mi_in = log_10(mi)
        for asp in asps:
            for n in ns:
                # Prepare model input for all xsize and theta combinations
                data_list = []
                model_in_all = []
                model_in_nolog = []
                # Loop over all xsize and theta combinations to create batch input
                for xsize in xsizes:
                    xsize_in = log_10(xsize)
                    for theta in thetas:
                        model_in = [mr, mi_in, asp, n, xsize_in, theta]
                        model_in_all.append(model_in)
                # Convert model_in_all to numpy array (batch input)
                model_in_all = np.array(model_in_all, dtype='float32') 

                # Predict using the entire batch input for each model
                p11 = model_p11.predict(model_in_all).flatten() 
                dp12 = model_dp12.predict(model_in_all).flatten()
                dp22 = model_dp22.predict(model_in_all).flatten()
                dp33 = model_dp33.predict(model_in_all).flatten()
                dp44 = model_dp44.predict(model_in_all).flatten()
                dp43 = model_dp43.predict(model_in_all).flatten()
