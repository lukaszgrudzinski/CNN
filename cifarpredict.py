# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
toshow=0                   #determines starting point of visualization array
if len(sys.argv)==2:       #determines starting point of visualization array
	toshow=int(sys.argv[1])     #determines starting point of visualization array

print(tf.__version__)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
	
mnist = unpickle("cifar-10-batches-py/data_batch_1")

train_images=mnist[b'data']
train_labels=mnist[b'labels']

labelnames = unpickle("cifar-10-batches-py/batches.meta")		
class_names=labelnames[b'label_names']

  
train_images = train_images / 255.0
#test_images = test_images / 255.0

def unflatten(flat):
	licznik=0
	imgToShow=np.zeros([32,32])
	for i in range(0,1024):
		imgToShow[licznik,i%32]=flat[i]
		if i%32==31:
			licznik+=1
	return imgToShow

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
	
	#plt.subplot(5,5,i+1)
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
train_images_sel=np.zeros([1000,32,32])                         #selekcja kotow
j=0                                                             #selekcja kotow
for i in range(10000):                                          #selekcja kotow
	if train_labels[i]==3 and j<1000:                           #selekcja kotow
		train_images_sel[j]=unflatten(train_images[i])          #selekcja kotow
		j+=1                                                    #selekcja kotow

train_full=train_images_sel.copy()#kopia pelnych kotow
before=train_images_sel.copy()

from random import *                                            #robienie dziur 
for i in range(1000):                                           #robienie dziur 
		x=randint(4,28)                                         #robienie dziur 
		r=randint(3,4)                                          #robienie dziur 
		for j in range(32):                                     #robienie dziur 
			for k in range(32):                                 #robienie dziur 
				if j>x-r and j<x+r and k>x-r and k<x+r:         #robienie dziur 
					train_images_sel[i,j,k]=-1                  #robienie dziur 
withHole=train_images_sel.copy()
#show25first(train_images_sel)		
#show25first(train_full)



model = keras.models.load_model('nonconv.h5')
model.summary()



trainy=np.zeros([1000,1024])
trainx=np.zeros([1000,1024])


for i in range(1000):
	trainy[i]=myflatten(train_full[i])
	trainx[i]=myflatten(train_images_sel[i])
abc=trainx.copy()

					
test_predictions = model.predict(trainy)
#show25first(unflatten(test_predictions))
#print(test_predictions.shape)
img_result=np.zeros([1000,32,32])
for i in range(1000):

	img_result[i]=unflatten(test_predictions[i])
	

compare(before,withHole,img_result,toshow)
print(img_result[40])

