%matplotlib inline 
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
from matplotlib import pyplot as plt
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
nb_epochs=3

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
	# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# build the model

hist=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epochs, batch_size=1000)
# Final evaluation of the model
score, acc = model.evaluate(X_test, y_test, verbose=0)
#print("CNN Error: %.2f%%" % (100-scores[1]*100))
print("Accuracy: %.2f%%" % (acc*100))
print("Test score: %.2f%%" % (score*100))

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(nb_epochs)
plt.plot(xc,train_acc)
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')

Train on 60000 samples, validate on 10000 samples
Epoch 1/3
60000/60000 [==============================] - 232s 4ms/step - loss: 0.4987 - acc: 0.8633 - val_loss: 0.1929 - val_acc: 0.9437
Epoch 2/3
60000/60000 [==============================] - 232s 4ms/step - loss: 0.1603 - acc: 0.9536 - val_loss: 0.1059 - val_acc: 0.9688
Epoch 3/3
60000/60000 [==============================] - 231s 4ms/step - loss: 0.1017 - acc: 0.9709 - val_loss: 0.0777 - val_acc: 0.9764
Accuracy: 97.64%
Test score: 7.77%

<matplotlib.text.Text at 0x7f6601f0e390>

