# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
	
mnist = unpickle("../cifar-10-batches-py/data_batch_1")
train_images=mnist[b'data']

  
train_images = train_images / 255.0
#test_images = test_images / 255.0
from random import * 
def makeholes(array):
	for i in range(array.shape[0]):                    		        #robienie dziur 
			x=randint(4,28)                                         #robienie dziur 
			r=randint(3,4)                                          #robienie dziur 
			for j in range(32):                                     #robienie dziur 
				for k in range(32):                                 #robienie dziur 
					if j>x-r and j<x+r and k>x-r and k<x+r:         #robienie dziur 
						array[i,j,k]=-1                 			#robienie dziur 
	return array

def unflatten(flat):
	licznik=0
	imgToShow=np.zeros([32,32])
	for i in range(0,1024):
		imgToShow[licznik,i%32]=flat[i]
		if i%32==31:
			licznik+=1
	return imgToShow
def unflattenwholearray(array):
	a=array.shape[0]
	result=np.zeros([a,32,32])
	for i in range(array.shape[0]):
		result[i]=unflatten(array[i])
	return result
def myflatten(img):
	#for 32x32
	flat=np.zeros(1024)
	x=0
	for i in range(32):
		for j in range(32):
			flat[x]=img[i,j]
			x+=1
	return flat
def show25first(array):
	plt.figure(figsize=(13,13))                                     
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid('off')
		plt.imshow(array[i],interpolation="nearest")
	plt.show()
def showSingle(img):
	plt.figure(figsize=(3,3))                                     
	plt.xticks([])
	plt.yticks([])
	plt.grid('off')
	plt.imshow(img)
	plt.show()
def compare(before,withHole,after,start):
	plt.figure(figsize=(13,13))                                     
	for i in range(5):
		plt.subplot(5,3,i*3+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid('off')
		plt.imshow(before[i+start],vmin=0, vmax=1)
		
		plt.subplot(5,3,i*3+2)
		plt.xticks([])
		plt.yticks([])
		plt.grid('off')
		plt.imshow(withHole[i+start],vmin=0, vmax=1)
		
		plt.subplot(5,3,i*3+3)
		plt.xticks([])
		plt.yticks([])
		plt.grid('off')
		plt.imshow(after[i+start],vmin=0, vmax=1)
	plt.show()
def turnGray(array):
	gray=np.zeros([10000,1024])
	for i in range(10000):
		for j in range(1024):
			gray[i,j]=0.299*array[i,j]+0.587*array[i,j+1024]+0.114*array[i,j+2048]
	return gray

train_images=turnGray(train_images)
full_images=unflattenwholearray(train_images)
with_holes=makeholes(full_images.copy())
compare(with_holes,full_images,full_images,5)
full_images=np.expand_dims(full_images, axis=3)
with_holes=np.expand_dims(with_holes, axis=3)

model = keras.models.load_model('model.h5')
model.summary()
y=model.predict(with_holes)

compare(full_images[:,:,:,0],with_holes[:,:,:,0],y[:,:,:,0],3)
