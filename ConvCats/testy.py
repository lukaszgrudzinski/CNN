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
labelnames = unpickle("../cifar-10-batches-py/batches.meta")		
class_names=labelnames[b'label_names']
  
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
train_images=unflattenwholearray(train_images)
#at this point train_images is 10k gray images
#mnist = unpickle("../cifar-10-batches-py/data_batch_2")
#train_images_temp=mnist[b'data']
#train_images_temp=turnGray(train_images_temp)
#train_images_temp=unflattenwholearray(train_images_temp)
train_array=np.empty((1,10000,32,32))
train_array[0]=train_images.copy()
#train_array[1]=train_images_temp.copy()
#mnist = unpickle("../cifar-10-batches-py/data_batch_3")
#train_images_temp=mnist[b'data']
#train_images_temp=turnGray(train_images_temp)
#train_images_temp=unflattenwholearray(train_images_temp)
#train_array[2]=train_images_temp
#mnist = unpickle("../cifar-10-batches-py/data_batch_4")
#train_images_temp=mnist[b'data']
#train_images_temp=turnGray(train_images_temp)
#train_images_temp=unflattenwholearray(train_images_temp)
#train_array[3]=train_images_temp
#mnist = unpickle("../cifar-10-batches-py/data_batch_5")
#train_images_temp=mnist[b'data']
#train_images_temp=turnGray(train_images_temp)
#train_images_temp=unflattenwholearray(train_images_temp)
#train_array[4]=train_images_temp

test_array=train_array.copy()
for i in range(1):
	test_array[i]=makeholes(test_array[i])

trainx=np.expand_dims(train_images, axis=3)
trainy=np.expand_dims(test_array[0], axis=3)
print(trainy.shape)

#LEARNING TIME!
def build_model():
  model = keras.Sequential([
	keras.layers.Conv2D(16, (3,3), padding='same',data_format="channels_last", input_shape=(32,32,1)),
	keras.layers.AveragePooling2D(pool_size=(1,16), strides=None, padding='valid', data_format="channels_last"),
    keras.layers.Dense(32, activation=tf.tanh),
    keras.layers.Dense(32,activation=tf.tanh)
  ])

  sgd=keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
  model.compile(loss='mean_squared_error', optimizer=sgd)
  return model

model = build_model()
model.summary()

EPOCHS = 100

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 1 == 0: print('')
    print(epoch*100/EPOCHS,'%')
    print('.', end='')

history = model.fit(trainx, trainy, epochs=EPOCHS,
                  validation_split=0.2, verbose=0,
                  callbacks=[PrintDot()])
model.save('model.h5')
