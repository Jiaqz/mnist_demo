# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:40:30 2018

@author: acer
"""

import numpy as np
import matplotlib.pyplot as plt
from mnist_data import *
#get data
X_train=np.array(fetch_traingset()['images'])
X_train=X_train-np.mean(X_train,axis=0)
Y_train0=np.array(fetch_traingset()['labels'])
X_test=np.array(fetch_testingset()['images'])
X_test=X_test-np.mean(X_test,axis=0)
Y_test0=np.array(fetch_testingset()['labels'])

#new label
y0=np.zeros((60000,10))
index1=np.arange(60000)
index2=Y_train0
y0[index1,index2]=1
Y_train=y0
y1=np.zeros((10000,10))
index1=np.arange(10000)
index2=Y_test0
y1[index1,index2]=1
Y_test=y1

#initialization
W1=np.random.randn(784,200)/np.sqrt(392)
W2=np.random.randn(200,50)/np.sqrt(100)
W3=np.random.randn(50,20)/np.sqrt(25)
W4=np.random.randn(20,10)/np.sqrt(10)
lr=0.001
hyper=0.01
function='Relu'

def forward(x,W1,W2,W3,W4,function,hyper):
	L1=activate(np.dot(x,W1),function,hyper)
	L2=activate(np.dot(L1,W2),function,hyper)
	L3=activate(np.dot(L2,W3),function,hyper)
	y=activate(np.dot(L3,W4),function,hyper)
	return y

def activate(x,function,hyper):
	if(function=='Relu'):
		return np.where(x<=0,0,x)
	elif(function=='Sigmoid'):
		return 1/(1+np.exp(-x))
	elif(function=='Tanh'):
		return np.tanh(x)
	elif(function=='Leakly Relu'):
		return np.where(x<=0,hyper*x,x)
	elif(function=='Elu'):
		return np.where(x<=0,hyper*(np.exp(x)-1),x)

def gradient(x,function,hyper):
	if(function=='Relu'):
		return np.where(x<=0,0,1)
	elif(function=='Sigmoid'):
		return x*(1-x)
	elif(function=='Tanh'):
		return 1-x*x
	elif(function=='Leakly Relu'):
		return np.where(x<=0,hyper,1)
	elif(function=='Elu'):
		return np.where(x<=0,hyper*np.exp(x),1)


def update(X_train,Y_train,W1,W2,W3,W4,function,hyper,lr):
	L1=activate(np.dot(X_train,W1),function,hyper)
	L2=activate(np.dot(L1,W2),function,hyper)
	L3=activate(np.dot(L2,W3),function,hyper)
	y=activate(np.dot(L3,W4),function,hyper)
	y_delta=(Y_train-y)*gradient(y,function,hyper)
	L3_delta=y_delta.dot(W4.T)*gradient(L3,function,hyper)
	L2_delta=L3_delta.dot(W3.T)*gradient(L2,function,hyper)
	L1_delta=L2_delta.dot(W2.T)*gradient(L1,function,hyper)
	W4+=L3.T.dot(y_delta)*lr 
	W3+=L2.T.dot(L3_delta)*lr 
	W2+=L1.T.dot(L2_delta)*lr 
	W1+=X_train.T.dot(L1_delta)*lr
	return W1,W2,W3,W4

#def BatchNormalization(data):
def classify(x):
	x=np.argmax(x,axis=1)
	return x

def train(batch_size=50,w1=W1,w2=W2,w3=W3,w4=W4,iteration=1200):
	for i in range(iteration):
		x_train=X_train[i*batch_size:(i+1)*batch_size]
		y_train=Y_train[i*batch_size:(i+1)*batch_size]
		w1,w2,w3,w4=update(x_train,y_train,w1,w2,w3,w4,function,hyper,lr)
	return w1,w2,w3,w4

train_accuracy=[]
test_accuracy=[]
#train
for i in range(100):
	W1,W2,W3,W4=train(100,W1,W2,W3,W4,600)
	y=forward(X_train,W1,W2,W3,W4,function,hyper)
	y=classify(y)
	train_accuracy.append(1-np.sum(Y_train0!=y)/60000.0)
	y=forward(X_test,W1,W2,W3,W4,function,hyper)
	y=classify(y)
	test_accuracy.append(1-np.sum(Y_test0!=y)/10000.0)
	#print('epoch'+'{:2d}'.format(i+1)+' Train accuracy: ',1-np.sum(Y_train!=y)/60000.0)


#final accuracy
y_pre=forward(X_test,W1,W2,W3,W4,function,hyper)
y_pre=classify(y_pre)
print('Test accuracy: ',1-np.sum(Y_test0!=y_pre)/10000.0)
y_pre=forward(X_train,W1,W2,W3,W4,function,hyper)
y_pre=classify(y_pre)
print('Train accuracy: ',1-np.sum(Y_train0!=y_pre)/60000.0)

#plotting
index=np.arange(100)
plt.plot(index,train_accuracy,color="blue",linestyle="-",label='train_accuracy')
plt.plot(index,test_accuracy,color="red",linestyle="-",label='test_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

