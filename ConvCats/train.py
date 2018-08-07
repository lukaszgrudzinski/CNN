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
train_labels=mnist[b'labels']

labelnames = unpickle("../cifar-10-batches-py/batches.meta")		
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
	gray=np.zeros(len(array)/3,1024)
	for i in range(len(array)/3):
		for j in range(1024):
			gray[i,j]=0.299*array[i,j]+0.587*array[i,j+1024]+0.114*array[i,j+2048]
	return gray
train_full=train_images.copy()#kopia pelnych kotow
print(train_images.shape)
show25first(train_images)
from random import *                                            #robienie dziur 
for i in range(train_images.shape[0]):                                           #robienie dziur 
		x=randint(4,28)                                         #robienie dziur 
		r=randint(3,4)                                          #robienie dziur 
		for j in range(32):                                     #robienie dziur 
			for k in range(32):                                 #robienie dziur 
				if j>x-r and j<x+r and k>x-r and k<x+r:         #robienie dziur 
					train_images[i,j,k]=-1                      #robienie dziur 

withHole=train_images_sel.copy()
print(train_images.shape)
train_images=np.expand_dims(train_images_sel, axis=0)
print(train_images.shape)
def build_model():
  model = keras.Sequential([
  	#keras.layers.Conv2D(256, 3, strides=(1, 1), padding='valid', data_format="channels_first", dilation_rate=(1, 1),
	#					activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
	#					kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(32,32)),
	keras.layers.Conv2D(16, (3, 3), padding='same',data_format="channels_first", input_shape=(32,32)),
	keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format="channels_first"),
    keras.layers.Dense(128, activation=tf.tanh),

    keras.layers.Dense(1000,activation=tf.tanh)
  ])

 
  sgd=keras.optimizers.SGD(lr=5, momentum=0.0, decay=0.0, nesterov=False)
  model.compile(loss='mean_squared_error', optimizer=sgd)
  return model

model = build_model()
model.summary()

EPOCHS = 5

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 1 == 0: print('')
    print(epoch*100/EPOCHS,'%')
    print('.', end='')

trainy=np.zeros([1000,1024])
trainx=np.zeros([1000,1024])


#for i in range(1000):
#	trainy[i]=myflatten(train_full[i])
#	#trainx[i]=myflatten(train_images_sel[i])
trainy=train_full
trainy=np.expand_dims(trainy, axis=0)
abc=trainx.copy()
print(trainy.shape)
history = model.fit(train_images_sel, trainy, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

					
test_predictions = model.predict(abc)
img_result=np.zeros([1000,32,32])
for i in range(1000):

	img_result[i]=unflatten(test_predictions[i])
	

compare(before,withHole,img_result)
model.save('model.h5')
print(img_result[40])

