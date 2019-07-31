import numpy as np
from keras import optimizers
import random
import csv
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from pylab import subplot
import pandas as pd
from PIL import Image
from keras.utils import to_categorical
import os


model = Sequential()
model.add(Conv2D(8, (7, 7), input_shape=(512, 512, 1), kernel_initializer="glorot_normal"))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (4, 4), kernel_initializer="glorot_normal"))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3), kernel_initializer="glorot_normal"))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(64, kernel_initializer="glorot_normal"))
model.add(Activation('tanh'))
model.add(Dropout(0.2))

model.add(Dense(32, kernel_initializer="glorot_normal"))
model.add(Activation('tanh'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.00001, amsgrad=True),
              metrics=["categorical_accuracy"])


csv_files_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/csv'
train_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/train'
val_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/val'
val_csv_filename = 'val.csv'
train_csv_filename = 'train.csv'

batch_size_train= 400
batch_size_val = 400
num_epochs = 5


#################### Shuffle the csv file ####################################

def load_data (batch, image_dir):
    image_array = []
    label_array = []
    for b in batch:
        image_array.append(np.asarray(Image.open(image_dir + '/' + b[0])))
        label_array.append(b[1])
    labels = np.array(label_array)
    images = np.array(image_array)
    return images, labels


def load_csv (csv_files_dir, csv_filename) :
    csv_rows = []
    with open(csv_files_dir + '/' + csv_filename, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            csv_rows.append(row)
    #print('csv_rows[-1] : ', csv_rows[-1], len(csv_rows))
    return csv_rows

def save_csv (csv_file_dir, csv_filename, csv_list):
    csvData = []
    with open(csv_file_dir + '/' + 'new_' + csv_filename,'w') as csv_File:
        writer = csv.writer(csv_File)
        for row in csv_list:
            csvData.append(row)
        writer.writerows(csvData)
    csv_File.close()

def save_model(model, dir, epoch=None, val_loss=None):
    # serialize model to JSON
    model_json =  model.to_json()

    model_directory = "%s/save_model/" % (dir)
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    model_filename_json =  "%s/save_model/updated_model_epoch_%d_validation_loss_%0.3f.json" % (dir, epoch, val_loss)
    print(model_filename_json)

    model_filename_h5 = "%s/save_model/updated_model_epoch_%d_validation_loss_%0.3f.h5" % (dir, epoch, val_loss)

    with open(model_filename_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filename_h5)
    print("Saved model to disk")


train_imagename_label_pairs = load_csv(csv_files_dir, train_csv_filename)
val_imagename_label_pairs = load_csv(csv_files_dir, val_csv_filename)

#-----------------------------------train data ---------------------------------------------------
train_acc = []
train_loss=[]
random.shuffle(train_imagename_label_pairs)
[train_images, train_labels] = load_data(train_imagename_label_pairs, train_image_file_dir)

image_shape = train_images.shape
print("train_images.shape" , train_images.shape)
train_images2 = np.reshape(train_images, [-1, image_shape[1], image_shape[2], 1])
train_labels1 = to_categorical(train_labels)
train_labels2 = np.reshape(train_labels1, [batch_size_train, 2])

#--------------------------------validation data-------------------------------------------
val_acc=[]
val_loss =[]
random.shuffle(val_imagename_label_pairs)
[val_images, val_labels] = load_data(val_imagename_label_pairs, val_image_file_dir)

image_shape = val_images.shape
print("val_images.shape" , val_images.shape)
val_images2 = np.reshape(val_images, [-1, image_shape[1], image_shape[2], 1])
val_labels1 = to_categorical(val_labels)
val_labels2 = np.reshape(val_labels1, [batch_size_val, 2])

history = model.fit(x=train_images2, y=train_labels2,  epochs = num_epochs ,batch_size=10, validation_data=[val_images2, val_labels2], shuffle=True, verbose=2)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
