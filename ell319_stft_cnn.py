import numpy as np
import matplotlib.pyplot as plt
import argparse
#from libsvm import svmutil
from scipy import signal
#from stockwell import st
import cv2
import os
#import libsvm
# import csv
import pandas as pd
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input,LSTM,Bidirectional 
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
os.environ["CUDA_VISIBLE_DEVICES"]="0"

t1 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--X1", default="data/X_train.npy", type=str, help="Read Training data content from file")
parser.add_argument("--Y1", default="data/Y_train.npy", type=str, help="Read Training labels content from file")
parser.add_argument("--X2", default="data/X_test.npy", type=str, help="Read testing Data content from file")
parser.add_argument("--Y2", default="data/Y_test.npy", type=str, help="Read testing labels content from file")
#parser.add_argument("--Z", default="E:/IIT DELHI/SEMESTER-5/ELL409/ASSIGNMENTS/A2/result.csv", type=str, help="Write result to this file")
args = parser.parse_args()

X1 = args.X1
Y1 = args.Y1
X2 = args.X2
Y2 = args.Y2
#Z = args.Z


# Reading the training Data from binary to decimal
xtrain = np.load(X1)
ytrain = np.load(Y1)
#ytrain = ytrain.astype('int32')
# ytrain = ytrain.reshape(-1,1)
xtest = np.load(X2)
ytest = np.load(Y2)
# ytest = ytest.reshape(-1,1)
# print("Training Data", xtrain)

dimtrain = xtrain.shape
n1 = dimtrain[0];  # No of samples
n2 = dimtrain[1];
n3 = dimtrain[2];

dimtest = xtest.shape
n4 = dimtest[0]
n5 = dimtest[1]
n6 = dimtest[2]


#Using STFT transform to generate TF-plots and converting the images to pixel data for training data

print('Training data')

cnninputtrain = np.zeros((n1,16,11,6))
for k in range(n1):
    for i in range(6):
        x = xtrain[k, :,i]
        f,t,zxx = signal.stft(x, nperseg=31)
        zmatrix = np.abs(zxx)
        zxx_max = np.max(zmatrix)
        zmatrix = zmatrix/zxx_max
        
        cnninputtrain[k,:,:,i] = zmatrix



print('Training data done')




#Using STFT transform to generate TF-plots and converting the images to pixel data for testing data

cnninputtest = np.zeros((n4,16,11,6))
for r in range(n4):
    for p in range(6):
        x = xtest[r, :,p]
        f,t,zxx = signal.stft(x, nperseg=31)
        zx = np.abs(zxx)
        max1 = np.max(zx)
        zx = zx/max1
        cnninputtest[r,:,:,p] = zx
				
print('Testing data done')

t2 = time.time()

print(t2-t1)


i1 = Input(shape=(16,11,6))
x1 = Conv2D(64,(3,3),padding='same')(i1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64,(3,3),padding='same')(x1)
x1 = MaxPooling2D()(x1)
x1 = Conv2D(64,(3,3),padding='same')(x1)
x1 = MaxPooling2D()(x1)
x1 = Flatten()(x1)
x1 = Dropout(0.5)(x1)
output = Dense(26, activation='softmax')(x1)

model = Model(inputs=i1, outputs=output)

print(model.summary())
#plot_model(model, to_file='shared_input_layer1.png')

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_accuracy', verbose=0, patience=5)
model.fit(cnninputtrain, y=to_categorical(ytrain),validation_split=0.2,epochs=50, batch_size=32,verbose=1,callbacks=[es])

predictions = model.predict(cnninputtest)
Y_pred = np.argmax(np.asarray(predictions),axis=1)

acc = accuracy_score(ytest,Y_pred)
print(acc)





