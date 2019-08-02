import numpy as np
import tensorflow as tf
from keras import optimizers
import csv
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from PIL import Image
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import time
from numpy import random
import os


#-------------------------Tensorboard Class -------------------------------#
class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='/logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

#-------------------------Model -------------------------------#
Name = 'updated_model-{}'.format(int(time.time()))
tensorboard = TrainValTensorBoard(log_dir='logs/{}'.format(Name))

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

#-------------------------Constants -------------------------------#

csv_files_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/csv'
train_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/train'
val_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/val'
val_csv_filename = 'val.csv'
test_csv_filename = 'test.csv'
train_csv_filename = 'train.csv'

batch_size_train= 400
batch_size_val = 400
num_epochs = 5


#-------------------------Shuffle the csv file -------------------------------#

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

def save_model(model):
    # serialize model to JSON
    model_json =  model.to_json()

    model_directory = "/home/mashids/PycharmProjects/data_processing/save_model_new/"
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    model_filename_json =  "/home/mashids/PycharmProjects/data_processing/save_model_new/"+Name+ ".json"
    print(model_filename_json)

    model_filename_h5 = "/home/mashids/PycharmProjects/data_processing/save_model_new/" +Name+ ".h5"

    with open(model_filename_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filename_h5)
    print("Saved model to disk")

a=[]
b=[]
train_imagename_label_pairs = load_csv(csv_files_dir, train_csv_filename)
print("train_imagename_label_pairs: ", len(train_imagename_label_pairs))
val_imagename_label_pairs = load_csv(csv_files_dir, val_csv_filename)
print("val_imagename_label_pairs",len(val_imagename_label_pairs))

for r, d, f in os.walk(train_image_file_dir):
    for lll in train_imagename_label_pairs :
      for file in f :
         if lll[0] == str(file) :
             a.append(1)

for r, d, f in os.walk(val_image_file_dir):
    for lll in val_imagename_label_pairs :
      for file in f :
         if lll[0] == str(file) :
             b.append(1)
print("len(a)" , len(a))
print("len(b)" , len(b))
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


history = model.fit(x=train_images2, y=train_labels2,  epochs = num_epochs ,batch_size=10, validation_data=[val_images2, val_labels2],
                    verbose=2 , shuffle=True,
                    callbacks=[tensorboard])

dir = os.getcwd()
save_model(model)

