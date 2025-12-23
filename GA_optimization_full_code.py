import pygad
import tensorflow as tf
import sympy as sym
import numpy as np
import math
from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import fsolve
import math
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle as pickle
from tensorflow.keras import backend
import tensorflow.compat.v1 as tf
import matplotlib
import cv2
import sys
import numpy
import scipy.io as sio
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters
import random
import csv
from PIL import Image
tf.reset_default_graph()
import os


ma = 0
def fitness_func(ga_instance, solution, solution_idx):
    global ma
    import numpy as np
    import cv2
    P0 = solution[0]
    P1 = solution[1]
    P2 = solution[2]
    P3 = solution[3]
    P4 = solution[4]
    P5 = solution[5]
    P6 = solution[6]
    P7 = solution[7]
    P8 = solution[8]
    zval = [P0, P1, P2, P3, P4, P5, P6, P7, P8]
    #zval = zval1.tolist()
    print(zval)
    def lrelu(x, alpha=0.2, name=None):
        return tf.nn.leaky_relu(x, alpha, name=name)
    def relu(x, name=None):
        return tf.nn.relu(x, name=name)
    tf.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    sess = tf.Session()
    #/home/ravikiranb/ConvLSTM_microstructure/model
    imported_graph = tf.train.import_meta_graph('/model_3-14999.meta')
    imported_graph.restore(sess, tf.train.latest_checkpoint('/'))
    print("done1")
    T_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='')
    D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
    G_vars = [var for var in T_vars if var.name.startswith('generator')]
    W_vars = [var for var in G_vars if var.name.endswith('kernel:0')]
    bias_vars = [var for var in G_vars if var.name.endswith('bias:0')]
    gamma_vars = [var for var in G_vars if var.name.endswith('gamma:0')]
    beta_vars = [var for var in G_vars if var.name.endswith('beta:0')]
    mean_vars = [var for var in G_vars if var.name.endswith('moving_mean:0')]
    var_vars = [var for var in G_vars if var.name.endswith('moving_variance:0')]

    export_var = {}
    export_varb ={}
    export_varg ={}
    export_varbe ={}
    export_varm ={}
    export_varv ={}
    vars_vals =sess.run(G_vars)
    vars_vals_w =sess.run(W_vars)
    vars_vals_bias =sess.run(bias_vars)
    vars_vals_gamma =sess.run(gamma_vars)
    vars_vals_beta =sess.run(beta_vars)
    vars_vals_m =sess.run(mean_vars)
    vars_vals_v =sess.run(var_vars)
    #export_var=dict(zip(W_vars[:,0], vars_vals_w))
    for var, val in zip(W_vars, vars_vals_w):
        export_var[var.name] =  val
    for var, val in zip(bias_vars, vars_vals_bias):
        export_varb[var.name] =  val

    for var, val in zip(gamma_vars, vars_vals_gamma):
        export_varg[var.name] =  val 

    for var, val in zip(beta_vars, vars_vals_beta):
        export_varbe[var.name] =  val

    for var, val in zip(mean_vars, vars_vals_m):
        export_varm[var.name] =  val 

    for var, val in zip(var_vars, vars_vals_v):
        export_varv[var.name] =  val
    sess.close()
    weights = export_var
    biases  = export_varb
    gamma = export_varg
    beta = export_varbe
    mean = export_varm
    varience = export_varv
    # G(z)
    def generator(x, isTrain=True, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            # 1st hidden layer
            conv1 = tf.layers.conv2d_transpose(x, filter_num[0], [4, 4], strides=(2, 2), padding='same', name='conv1',
                                               kernel_initializer=tf.constant_initializer(G_conv1_weights),
                                               bias_initializer=tf.constant_initializer(G_conv1_biases))

            bn1 = tf.layers.batch_normalization(conv1, training=isTrain, name='bn1',
                                                gamma_initializer=tf.constant_initializer(G_BN1_gamma),
                                                beta_initializer=tf.constant_initializer(G_BN1_beta),
                                                moving_mean_initializer=tf.constant_initializer(G_BN1_moving_mean),
                                                moving_variance_initializer=tf.constant_initializer(G_BN1_moving_variance))

            relu1 = relu(bn1, name='relu1')

# 2nd hidden layer
            conv2 = tf.layers.conv2d_transpose(relu1, filter_num[1], [4, 4], strides=(2, 2), padding='same', name='conv2',
                                               kernel_initializer=tf.constant_initializer(G_conv2_weights),
                                               bias_initializer=tf.constant_initializer(G_conv2_biases))

            bn2 = tf.layers.batch_normalization(conv2, training=isTrain, name='bn2',
                                                gamma_initializer=tf.constant_initializer(G_BN2_gamma),
                                                beta_initializer=tf.constant_initializer(G_BN2_beta),
                                                moving_mean_initializer=tf.constant_initializer(G_BN2_moving_mean),
                                                moving_variance_initializer=tf.constant_initializer(G_BN2_moving_variance))
            relu2 = relu(bn2, name='relu2')

# 3rd hidden layer
            conv3 = tf.layers.conv2d_transpose(relu2, filter_num[2], [4, 4], strides=(2, 2), padding='same', name='conv3',
                                               kernel_initializer=tf.constant_initializer(G_conv3_weights),
                                               bias_initializer=tf.constant_initializer(G_conv3_biases))
            bn3 = tf.layers.batch_normalization(conv3, training=isTrain, name='bn3',
                                                gamma_initializer=tf.constant_initializer(G_BN3_gamma),
                                                beta_initializer=tf.constant_initializer(G_BN3_beta),
                                                moving_mean_initializer=tf.constant_initializer(G_BN3_moving_mean),
                                                moving_variance_initializer=tf.constant_initializer(G_BN3_moving_variance))
            relu3 = relu(bn3, name='relu3')

# 4th hidden layer
            conv4 = tf.layers.conv2d_transpose(relu3, filter_num[3], [4, 4], strides=(2, 2), padding='same', name='conv4',
                                               kernel_initializer=tf.constant_initializer(G_conv4_weights),
                                               bias_initializer=tf.constant_initializer(G_conv4_biases))
            bn4 = tf.layers.batch_normalization(conv4, training=isTrain, name='bn4',
                                                gamma_initializer=tf.constant_initializer(G_BN4_gamma),
                                                beta_initializer=tf.constant_initializer(G_BN4_beta),
                                                moving_mean_initializer=tf.constant_initializer(G_BN4_moving_mean),
                                                moving_variance_initializer=tf.constant_initializer(G_BN4_moving_variance))
            relu4 = relu(bn4, name='relu4')

            conv5 = tf.layers.conv2d_transpose(relu4, 1, [4, 4], strides=(2, 2), padding='same', name='conv5',
                                               kernel_initializer=tf.constant_initializer(G_conv5_weights),
                                               bias_initializer=tf.constant_initializer(G_conv5_biases))
            o = tf.nn.tanh(conv5, name='o')
            return o


    # loaG the weights
    tf.compat.v1.disable_eager_execution()
    #weight_path = './weights_steel.pickle'

    #with open(weight_path, 'rb') as f:
    #weights, biases, BNs = pickle.load(f,encoding='latin1')

    G_conv1_weights = weights['generator/conv1/kernel:0']
    G_conv2_weights = weights['generator/conv2/kernel:0']
    G_conv3_weights = weights['generator/conv3/kernel:0']
    G_conv4_weights = weights['generator/conv4/kernel:0']
    G_conv5_weights = weights['generator/conv6/kernel:0']

    G_conv1_biases = biases['generator/conv1/bias:0']
    G_conv2_biases = biases['generator/conv2/bias:0']
    G_conv3_biases = biases['generator/conv3/bias:0']
    G_conv4_biases = biases['generator/conv4/bias:0']
    G_conv5_biases = biases['generator/conv6/bias:0']

    G_BN1_gamma = gamma['generator/batch_normalization/gamma:0']
    G_BN1_beta = beta['generator/batch_normalization/beta:0']
    G_BN1_moving_mean = mean['generator/batch_normalization/moving_mean:0']
    G_BN1_moving_variance = varience['generator/batch_normalization/moving_variance:0']

    G_BN2_gamma = gamma['generator/batch_normalization_1/gamma:0']
    G_BN2_beta = beta['generator/batch_normalization_1/beta:0']
    G_BN2_moving_mean = mean['generator/batch_normalization_1/moving_mean:0']
    G_BN2_moving_variance = varience['generator/batch_normalization_1/moving_variance:0']

    G_BN3_gamma = gamma['generator/batch_normalization_2/gamma:0']
    G_BN3_beta = beta['generator/batch_normalization_2/beta:0']
    G_BN3_moving_mean = mean['generator/batch_normalization_2/moving_mean:0']
    G_BN3_moving_variance = varience['generator/batch_normalization_2/moving_variance:0']

    G_BN4_gamma = gamma['generator/batch_normalization_3/gamma:0']
    G_BN4_beta = beta['generator/batch_normalization_3/beta:0']
    G_BN4_moving_mean = mean['generator/batch_normalization_3/moving_mean:0']
    G_BN4_moving_variance = varience['generator/batch_normalization_3/moving_variance:0']

    print ("====> Weights loaded!")
    z_dim = (1, 3, 3, 1)
    filter_num = [112, 56, 28, 14] 
    z = tf.placeholder(tf.float32, z_dim, name='z')
    isTrain = tf.placeholder(dtype=tf.bool, name='isTrain')
    X = generator(z, isTrain)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    z_ = np.reshape(np.array(zval), z_dim)
    image = sess.run([X], feed_dict={z:z_, isTrain:False})[0]
    print(image[0].shape)
    plt.figure(0)
    gray=image[0, :, :, 0]
    val = filters.threshold_otsu(gray)
    plt.imshow(gray>val, cmap='gray')
    plt.axis('on')
#plt.savefig('RS_3000x_3000t_128_128_2.png')
    print ("====> image generated!")
    print ("DONE")

    image.shape
    gray=image[0]
    plt.axis('off')
    val = filters.threshold_otsu(gray)
    plt.imshow(gray<val, cmap='gray')
    plt.savefig('Pearlite_96_'+ str(ma) +'.png')
    print ("====> image generated!")
    print ("DONE")

    img = cv2.imread(r'Pearlite_96_'+ str(ma) +'.png',0)
     #######Croping the image with out brackground
    import cv2
    import numpy as np

# Load image, convert to grayscale, Gaussian blur, Otsu's threshold
    image = cv2.imread('Pearlite_96_'+ str(ma) +'.png')
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Obtain bounding rectangle and extract ROI
    x,y,w,h = cv2.boundingRect(thresh)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    ROI = original[y:y+h, x:x+w]

    # Add alpha channel
    b,g,r = cv2.split(ROI)
    alpha = np.ones(b.shape, dtype=b.dtype) * 50
    ROI = cv2.merge([b,g,r,alpha])
    cv2.imwrite('Pearlite_96_'+ str(ma) +'-b.png', ROI)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('image', image)
    #cv2.imshow('ROI', ROI)
    cv2.waitKey()


# In[3]:


    ############ Resizing the image to 128x128
    import cv2
    img = cv2.imread('Pearlite_96_'+ str(ma) +'-b.png', cv2.IMREAD_UNCHANGED)
    scale_percent = 26.1 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * (scale_percent) / 100)
    dim = (width, height)
# resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.resize(resized, (96, 96)) 
    cv2.imwrite('Pearlite_96_'+ str(ma) +'-b1.png', resized)


# In[4]:


####Converting to binary Image
    import cv2  

    img = cv2.imread('Pearlite_96_'+ str(ma) +'-b1.png', 2)  
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  
# converting to its binary form
    bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  
    #cv2.imshow("Binary", bw_img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Pearlite_96_'+ str(ma) +'-b2.png',bw_img)
    img = cv2.imread('Pearlite_96_'+ str(ma) +'-b2.png', cv2.IMREAD_UNCHANGED)
    
    cv2.imwrite('Pearlite_96_'+ str(ma) +'-b2.png',bw_img)
    os.remove('Pearlite_96_'+ str(ma) +'.png')
    os.remove('Pearlite_96_'+ str(ma) +'-b.png')
    os.remove('Pearlite_96_'+ str(ma) +'-b1.png')
    #os.remove('Pearlite_96_'+ str(ma) +'-b2.png')
    
############## Input fraction data generation ##########
    img_me_arr = []
    lok = img/255
    lok = 1-lok
    lok_2 = lok.mean(axis=(0, 1))
    for l in range(11):
        in_imag = np.full((96, 96), lok_2)
        img_me_arr.append(in_imag) 
    img_me_arr = np.array(img_me_arr)
###################### Geometry Input ##############
    im_ge = []
    for l in range(11):
        im_ge.append(lok) 
    im_ge = np.array(im_ge)

    img_dim2 =(96,96,2)
    ki = np.stack((im_ge, img_me_arr), axis=-1)
    In_geo = ki.reshape((1,)+(11,)+ img_dim2)
#################### ConvLSTM model loding ##############
    model = tf.keras.models.load_model(r'/model_convlstm_5000_V21_Conv2D_2_layers_3_more_training.h5')
    pred_lstm = model.predict(In_geo)

    vonMises_max_value = 1616.8
    vonMises_min_value = 20.1

    max_stress_value = 834.5454545454545
    min_stress_value = 550.5454545454545


    stress_vst_resc = ((pred_lstm[:,:,:,:,0]*(vonMises_max_value-vonMises_min_value))+vonMises_min_value)
    stress_st_resc = ((pred_lstm[:,:,:,:,1]*(max_stress_value-min_stress_value))+min_stress_value)

    mean_stress_yi = np.mean(stress_st_resc[0,0,:,:]) ###### Yield stress

    stress_ulti = [] 
    stress_all = [] 
    for re in range (0,11):
        mean_stress_1 = np.mean(stress_st_resc[0,re,:,:])
        stress_all.append(mean_stress_1)
    stress_all = np.array(stress_all)
    stress_all_max = np.max(stress_all) 

    stress_ulti = stress_all_max

    mean_stress_con = np.mean(pred_lstm[:,10,:,:,0])
    max_stress_con = np.max(pred_lstm[:,10,:,:,0])

    stress_con_Kt = (max_stress_con/mean_stress_con)
    
    mean_stress_yi_norm = (abs(mean_stress_yi-550))/(700-550)
    stress_ulti_norm = (abs(stress_ulti-680))/(850-680)
    stress_con_Kt_norm = (abs(stress_con_Kt - 1.0))/(2.5-1.0) 
    
    Yprediction = -stress_con_Kt_norm
    ma = ma+1
    #if 0.115<lok_2<0.5: 
    OF = Yprediction
    #else:
    #OF = 0
    print(lok_2, ma-1, mean_stress_yi, stress_ulti, stress_con_Kt, OF)
    return OF


parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

ga_instance = pygad.GA(num_generations=300,
                       sol_per_pop=10,
                       num_parents_mating=5,
                       num_genes= 9,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=[{'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}, {'low': -1.0, 'high': 1.0}],
                       gene_type= float,
                       save_solutions=True,
                       stop_criteria=["saturate_15"])

ga_instance.run()


# In[ ]:


ga_instance.plot_fitness()

solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
print("Solution", solution)
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


# In[ ]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


import matplotlib.pyplot

matplotlib.pyplot.figure()
matplotlib.pyplot.plot(ga_instance.best_solutions_fitness)
matplotlib.pyplot.tick_params(right= True,top= True,  direction='in')
matplotlib.pyplot.savefig('PyGAD_figure_Opti_min_stress_concetration.eps',bbox_inches='tight', format='eps', dpi=1200)
matplotlib.pyplot.show()






