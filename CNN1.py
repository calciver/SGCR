import numpy as np
from keras import optimizers
import random
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense


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

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, kernel_initializer="glorot_normal"))  # , activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('tanh'))
model.add(Dropout(0.2))

model.add(Dense(32, kernel_initializer="glorot_normal"))  # , activity_regularizer=regularizers.l2(0.01)))
model.add(Activation('tanh'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.00001, amsgrad=True),
              metrics=["categorical_accuracy"])

path1 = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/info'
path3 = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/train'
filename1 = 'val.csv'
filename2 = 'test.csv'
filename3 = 'train.csv'
filename = filename3
path_info = path1
path_image = path3

#################### Shuffle the csv file ####################################
inputm = []
with open(path_info + '/' + filename, newline='') as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    for row in reader:
        inputm.append(row)
random.shuffle(inputm)

csvData = []
l = inputm
with open(path_info + '/' + 'new_' + filename,'w') as csv_File:
    writer = csv.writer(csv_File)
    for ll in l:
        csvData.append(ll)
    writer.writerows(csvData)
csv_File.close()


inputm1 = []
with open(path_info + '/' + 'new_' + filename, newline='') as csvFile:
    reader = csv.reader(csvFile, delimiter=',')
    for row in reader:
        inputm1.append(row)
print('inputm1[-1] : ', inputm1[-1], len(inputm1))


#############################################################################################
############################### Creating batches ###########################################
def divide_chunks(listt, n):
    for i in range(0, len(listt), n):
        yield listt[i:i + n]


n = 100  # batch size

batch = list(divide_chunks(inputm1, n))
if len(l) % n != 0:
    halfbatch = batch[-1]
    batch.pop()
    i = n - len(halfbatch)
    print('i : ', i)
    for j in range(i):
        halfbatch.append(inputm[i])
    batch.append(halfbatch)

############################# Read png files from folders ######################################
import PIL
import numpy
from PIL import Image
from keras.utils import to_categorical

epochs = 5
for epoch in range(epochs):
    for k in range(5):  # len(batch)):
        image_array = []
        lable_array = []

        for b in batch[k]:
            image_array.append(numpy.asarray(PIL.Image.open(path_image + '/' + b[0])))
            lable_array.append(b[1])

        train_lable = np.array(lable_array)
        train_data = np.array(image_array)
        img_shape = train_data.shape
        train_data2 = np.reshape(train_data, [-1, img_shape[1], img_shape[2], 1])
        train_lable1 = to_categorical(train_lable)
        train_lable2 = np.reshape(train_lable1, [n, 2])
        history = model.fit(x=train_data2, y=train_lable2)

##############################################################################################


