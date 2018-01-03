import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
######################################################################################
# The execting Steps are:
sam@sam-pc:~/handw$  python conv.py
Using TensorFlow backend.
2018-01-03 20:55:53.963571: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
 - 260s - loss: 0.2310 - acc: 0.9344 - val_loss: 0.0828 - val_acc: 0.9744
Epoch 2/10
 - 233s - loss: 0.0737 - acc: 0.9781 - val_loss: 0.0465 - val_acc: 0.9842
Epoch 3/10
 - 231s - loss: 0.0532 - acc: 0.9839 - val_loss: 0.0427 - val_acc: 0.9859
Epoch 4/10
 - 222s - loss: 0.0401 - acc: 0.9878 - val_loss: 0.0407 - val_acc: 0.9868
Epoch 5/10
 - 222s - loss: 0.0337 - acc: 0.9893 - val_loss: 0.0347 - val_acc: 0.9884
Epoch 6/10
 - 226s - loss: 0.0276 - acc: 0.9916 - val_loss: 0.0308 - val_acc: 0.9896
Epoch 7/10
 - 224s - loss: 0.0232 - acc: 0.9927 - val_loss: 0.0357 - val_acc: 0.9879
Epoch 8/10
 - 224s - loss: 0.0207 - acc: 0.9934 - val_loss: 0.0333 - val_acc: 0.9882
Epoch 9/10
 - 229s - loss: 0.0168 - acc: 0.9944 - val_loss: 0.0312 - val_acc: 0.9901
Epoch 10/10
 - 231s - loss: 0.0144 - acc: 0.9958 - val_loss: 0.0322 - val_acc: 0.9901
CNN Error: 0.99%
