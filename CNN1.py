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


csv_files_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/info'
train_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/train'
val_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/val'
val_csv_filename = 'val.csv'
test_csv_filename = 'test.csv'
train_csv_filename = 'train.csv'


batch_size= 100
num_epochs = 2
train_acc_epoch=[]
train_loss_epoch=[]
val_acc_epoch = []
val_loss_epoch = []

#################### Shuffle the csv file ####################################
def prepare_batches(imagename_label_pairs, batch_size):
    batch_list = []
    for i in range(0, len(imagename_label_pairs), batch_size):

        batch = imagename_label_pairs[i:i + batch_size]
        if len(batch) % batch_size != 0:
            m = batch_size - len(batch)
            for j in range(m):
                batch.append(imagename_label_pairs[j])

        batch_list.append(batch)
    return batch_list

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

    model_filename_json =  "%s/save_model/model_epoch_%d_validation_loss_%0.3f.json" % (dir, epoch, val_loss)
    print(model_filename_json)

    model_filename_h5 = "%s/save_model/model_epoch_%d_validation_loss_%0.3f.h5" % (dir, epoch, val_loss)

    with open(model_filename_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filename_h5)
    print("Saved model to disk")


train_imagename_label_pairs = load_csv(csv_files_dir, train_csv_filename)
val_imagename_label_pairs = load_csv(csv_files_dir, val_csv_filename)

val_batch_list = prepare_batches(val_imagename_label_pairs, batch_size)

print(len(val_batch_list))

for epoch in range(num_epochs):
    random.shuffle(train_imagename_label_pairs) 
    train_batch_list = prepare_batches(train_imagename_label_pairs, batch_size)

#-----------------------------------train model ---------------------------------------------------

    accu = 0
    loss = 0

    for k in range(len(train_batch_list)):
        print("epoch : ", epoch , "training batch : ", k )

        [train_images, train_labels] = load_data(train_batch_list[k], train_image_file_dir)

        image_shape = train_images.shape
        train_images2 = np.reshape(train_images, [-1, image_shape[1], image_shape[2], 1])
        train_labels1 = to_categorical(train_labels)
        train_labels2 = np.reshape(train_labels1, [batch_size, 2])

        history = model.fit(x=train_images2, y=train_labels2, batch_size=batch_size)
        accu= accu + history.history['categorical_accuracy'][0]
        loss= loss+ history.history['loss'][0]


    accu = accu / len(train_batch_list)
    loss = loss / len(train_batch_list)

    train_acc_epoch.append(accu)
    train_loss_epoch.append(loss)

 #-------------------------------------validate model-----------------------------------------

    accu = 0
    loss = 0

    for k in range(1): #range(len(val_batch_list)):
        print("epoch : ", epoch, "validation batch : ", k)

        [val_images, val_labels] = load_data(val_batch_list[k], val_image_file_dir)

        image_shape = val_images.shape
        val_images2 = np.reshape(val_images, [-1, image_shape[1], image_shape[2], 1])
        val_labels1 = to_categorical(val_labels)
        val_labels2 = np.reshape(val_labels1, [batch_size, 2])

        history = model.evaluate(x=val_images2, y=val_labels2, batch_size=batch_size)
        accu = accu + history[0]
        loss = loss + history[1]

    accu = accu / batch_size
    loss = loss / batch_size
    val_acc_epoch.append(accu)
    val_loss_epoch.append(loss)

#------------------------save model, loss and accuracies-------------------------------------

dir = os.getcwd()
save_model(model, dir, epoch, val_acc_epoch[-1])

train_acc_filename = "train_accu.npy"
np.save(train_acc_filename, train_acc_epoch)

train_loss_filename = "train_loss.npy"
np.save(train_loss_filename, train_loss_epoch)

val_acc_filename = "val_accu.npy"
np.save(val_acc_filename, val_acc_epoch)

val_loss_filename = "val_loss.npy"
np.save(val_loss_filename, val_loss_epoch)

#------------------plot train and validation accuracies and losses---------------------


subplot(2,1,1)
plt.plot(train_acc_epoch)
plt.plot(val_acc_epoch)
plt.legend(['Train accuracy', 'Validation accuracy'], loc='upper left')
plt.title('accuracy')
#plt.show()

subplot(2,1,2)

plt.plot(train_loss_epoch)
plt.plot(val_loss_epoch)
plt.legend(['Train loss', 'Validation loss'], loc='upper left')
plt.title('loss')
plt.show()
