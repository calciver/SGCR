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
import matplotlib.pyplot as plt

#------------------------------------------Constants--------------------------------------------------------------------

csv_files_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/selected_csv'
train_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/selected_train'
val_image_file_dir = '/home/mashids/Documents/CNN_Weakly_supervised/resized_images/selected_val'
val_csv_filename = 'val.csv'
test_csv_filename = 'test.csv'
train_csv_filename = 'train.csv'

#Number of epochs
num_epochs = 1

#---------------------------------------Tensorboard Class---------------------------------------------------------------
#This class is used to merge traing accuracy and validation accuracy graphs in Tensorboard

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

#--------------------------------Preparing data functions --------------------------------------------------------------

#Gets csv files directory and csv filename and returns rows in csv files which are image names and lables---------------
def load_csv (csv_files_dir, csv_filename) :
    csv_rows = []
    with open(csv_files_dir + '/' + csv_filename, newline='') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            csv_rows.append(row)
    #print('csv_rows[-1] : ', csv_rows[-1], len(csv_rows))
    return csv_rows

#returns numpy arrays of images and labels for each batch---------------------------------------------------------------
def load_data (batch, image_dir):
    image_array = []
    label_array = []
    for b in batch:
        image_array.append(np.asarray(Image.open(image_dir + '/' + b[0])))
        label_array.append(b[1])
    labels = np.array(label_array)
    images = np.array(image_array)
    return images, labels






#ploting 4 different categories-----------------------------------------------------------------------------------------
def show_samples(input_image , input_pairs  ,image_dir):

    correct_pneumonia=[]
    incorrect_pneumonia=[]
    correct_normal=[]
    incorrect_normal=[]
    prediction = np.empty(0)
#model.predict returns 2 valuse which are predictions of each label
    pred =model.predict(x=input_image, batch_size=None , verbose=0, steps=None)
    prediction = np.append(prediction,pred)
    prediction_list = prediction.tolist()
    zero_labels_predicted = prediction_list[::2]
    one_labels_predicted = prediction_list[1::2]

    for i in range(len(input_pairs)) :

        if input_pairs[i][1] == '1' :
            if one_labels_predicted[i] > 0.50 :
                print("correct_pneumonia : " , input_pairs[i][0])
                correct_pneumonia.append(input_pairs[i][0])
            elif one_labels_predicted[i] <0.50 :
                incorrect_pneumonia.append(input_pairs[i][0])

        elif input_pairs[i][1] == '0' :
            if zero_labels_predicted[i] > 0.50 :
                correct_normal.append(input_pairs[i][0])
                print("correct normal : ", input_pairs[i][0])
            elif zero_labels_predicted[i] < 0.50 :
                incorrect_normal.append(input_pairs[i][0])

#ploting images
    if correct_pneumonia != [] :
        plt.subplot(2, 2, 1)
        img = plt.imread(image_dir +'/' + correct_pneumonia[0])
        plt.imshow(img, cmap='gray')
        plt.title('correct_pneumonia')

    if incorrect_pneumonia != [] :
        plt.subplot(2, 2, 2)
        img = Image.open(image_dir + '/' + incorrect_pneumonia[0])
        plt.imshow(img, cmap='gray')
        plt.title('incorrect_pneumonia')

    if correct_normal !=[] :
        plt.subplot(2, 2, 3)
        img = Image.open(image_dir + '/' + correct_normal[0])
        plt.imshow(img, cmap='gray')
        plt.title('correct_normal')

    if incorrect_normal != [] :
        plt.subplot(2, 2, 4)
        img = Image.open(image_dir + '/' + incorrect_normal[0])
        plt.imshow(img, cmap='gray')
        plt.title('incorrect_normal')

    plt.show()


#saving model-----------------------------------------------------------------------------------------------------------
def save_model(model):
    # serialize model to JSON
    model_json =  model.to_json()

    model_directory = "/home/mashids/PycharmProjects/data_processing/save_model_/"
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    model_filename_json =  "/home/mashids/PycharmProjects/data_processing/save_model_/"+Name+ ".json"
    print(model_filename_json)

    model_filename_h5 = "/home/mashids/PycharmProjects/data_processing/save_model_/" +Name+ ".h5"

    with open(model_filename_json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_filename_h5)
    print("Saved model to disk")



#-----------------------------------Model Architecture------------------------------------------------------------------

Name = 'updated_model-{}'.format(int(time.time()))
tensorboard = TrainValTensorBoard(log_dir='logs/{}'.format(Name))

#set up a sequential model
model = Sequential()
#first convolutional layer with 8 channels and kenel size of 7*7
model.add(Conv2D(8, (7, 7), input_shape=(512, 512, 1), kernel_initializer="glorot_normal"))
# tanh activation function
model.add(Activation('tanh'))
#polling layer using Maxpooling of window size 2*2
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

#Summarize Model and Visualize Model
model.summary()

#Configures the model for training using  existing loss function 'categorical_crossnetropy' .
#'Adam' optimizer with learning rate of 0.0005.
#and 'categorical_accuracy' as metrices to be evaluated by the model during training and testing.
model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, decay=0.00001, amsgrad=True),
              metrics=["categorical_accuracy"])


#-----------------------------------Preparing training data ------------------------------------------------------------

train_imagename_label_pairs = load_csv(csv_files_dir, train_csv_filename)
#shuffle images and labels foe each epoch
random.shuffle(train_imagename_label_pairs)
[train_images, train_labels ] = load_data(train_imagename_label_pairs, train_image_file_dir)

#number of training data size
batch_size_train= len(train_imagename_label_pairs)
image_shape = train_images.shape
train_images2 = np.reshape(train_images, [-1, image_shape[1], image_shape[2], 1])
train_labels1 = to_categorical(train_labels)
train_labels2 = np.reshape(train_labels1, [batch_size_train, 2])

#--------------------------------Preparing validation data--------------------------------------------------------------

val_imagename_label_pairs = load_csv(csv_files_dir, val_csv_filename)
random.shuffle(val_imagename_label_pairs)
[val_images, val_labels] = load_data(val_imagename_label_pairs, val_image_file_dir)

#number of training validation size
batch_size_val = len(val_imagename_label_pairs)
image_shape = val_images.shape
val_images2 = np.reshape(val_images, [-1, image_shape[1], image_shape[2], 1])
val_labels1 = to_categorical(val_labels)
val_labels2 = np.reshape(val_labels1, [batch_size_val, 2])

#----------------------------------------Main---------------------------------------------------------------------------


history = model.fit(x=train_images2, y=train_labels2,  epochs = num_epochs ,batch_size=10,
                    validation_data=[val_images2, val_labels2],
                     verbose=2 , shuffle=True, callbacks=[tensorboard])


#ploting 4 samples of each catagory : correct_pneumonia ,incorrect_pneumonia, correct_normal, incorrect_normal
#if you can not see some of these categories it means model did not predict any samples of that categories
show_samples(train_images2 ,train_imagename_label_pairs , image_dir = train_image_file_dir)


#Saving Model
dir = os.getcwd()
save_model(model)



