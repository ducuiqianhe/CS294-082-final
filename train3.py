import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import h5py
import numpy as np
import os

img_height, img_width = 480, 640

def save_bottlebeck_features():
    # datagen = ImageDataGenerator(rescale=1./255,
    #     rotation_range = 20,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)
    datagen = ImageDataGenerator(rescale=1./255)
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_height, img_width, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    model.load_weights("./model/vgg16.h5")
    print('Model loaded.')

    generator = datagen.flow_from_directory(
        './image/test',
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_test = model.predict_generator(generator, 1531)
    print (bottleneck_features_test.shape)
    np.save(open('./data3/bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
    print('Validation features predicted!')

    # generator = datagen.flow_from_directory(
    #     './image/validation',
    #     target_size=(img_height, img_width),
    #     batch_size=5,
    #     class_mode=None,
    #     shuffle=False)
    
    # bottleneck_features_validation = model.predict_generator(generator, 59)
    # print (bottleneck_features_validation.shape)
    # np.save(open('./data3/bottleneck_features_validation1.npy', 'wb'), bottleneck_features_validation)
    # print('Validation features predicted!')

    # print("Starting feature prediction for the training set")
    # generator = datagen.flow_from_directory(
    #     './image/train',
    #     target_size=(img_height,img_width),
    #     batch_size=5,
    #     class_mode=None,
    #     shuffle=False)
    # bottleneck_features_train = model.predict_generator(generator, 400)
    # np.save(open('./data3/bottleneck_features_train1.npy', 'wb'), bottleneck_features_train)
    # print('Training features predicted! Starting feature prediction for the validation set')

def top_model0(train_shape):
    model = Sequential()
    model.add(Flatten(input_shape=train_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model

def top_model1(train_shape):
    model = Sequential()
    model.add(Flatten(input_shape=train_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


def top_model2(train_shape):
    model = Sequential()
    model.add(Flatten(input_shape=train_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


def top_model3(train_shape):
    model = Sequential()
    model.add(Flatten(input_shape=train_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model

def top_model4(train_shape):
    model = Sequential()
    model.add(Flatten(input_shape=train_shape))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.SGD(lr=1e-5, momentum=0.9, nesterov=True), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model
    

def train_top_model():
    print ("phase 2")
    train_data = np.load(open('./data3/bottleneck_features_train1.npy','rb'))
    print (train_data.shape)
    train_labels = np.array([0] * 738 + [1] * 1262)

    validation_data = np.load(open('./data3/bottleneck_features_validation1.npy', 'rb'))
    print (validation_data.shape)
    validation_labels = np.array([0] * 109 + [1] * 186)

    model = top_model4(train_data.shape[1:])
    # checkpoint
    checkpath = "./newnewModel/check5"
    if not os.path.exists(checkpath):       
        os.makedirs(checkpath)
    filepath = checkpath + "/weights" + "-{epoch:02d}-{acc:.5f}-{val_acc:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]

    # model.load_weights("./newnewModel/check0.1/weights-150-0.91186.hdf5")

    model.fit(train_data, train_labels,
              epochs=400, batch_size=16,
              validation_data=(validation_data, validation_labels), 
              callbacks=callbacks_list)
    model.save_weights("./newnewModel/model1.h5")



def fine_tune(batch_size):
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_height, img_width, 3)))

    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    print(model.summary())

    # load the weights of the VGG16 networks
    model.load_weights("./model/vgg16.h5")
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    # top_model = Sequential()
    # top_model.add(Flatten(input_shape=model.output_shape[1:]))
    # top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    # top_model.add(Dense(1, activation='sigmoid')) 


    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))

    top_model.add(Dense(4, activation='relu'))
    top_model.add(Dense(1, activation='sigmoid'))
    print(top_model.summary())
    

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights("./newnewModel/check4/weights-93-0.95900-0.91525.hdf5")
    # test_model = load_model("./newnewModel/check5/weights-52-0.96000-0.91186.hdf5")

    # print(test_model.summary())
    # add the model on top of the convolutional base
    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=0.00002, momentum=0.9),
                  metrics=['acc'])

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rotation_range = 30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest',
            horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            './image/train',  # this is the target directory
            target_size=(480, 640),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            './image/validation',
            target_size=(480, 640),
            batch_size=batch_size,
            class_mode='binary')

    # checkpoint
    checkpath = "./newModel/checkfull"
    if not os.path.exists(checkpath):       
        os.makedirs(checkpath)
    filepath = checkpath + "/weights4" + "-{epoch:02d}-{val_acc:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False)
    callbacks_list = [checkpoint]

    # model.load_weights("./VGG16Tuned1.h5")
    # with tf.device('/cpu:0'):
    model.fit(
            train_generator,
            steps_per_epoch=2000 // batch_size,
            epochs=2,
            validation_data=validation_generator,
            validation_steps=295 // batch_size, 
            callbacks=callbacks_list)
    model.save_weights('./VGG16Tuned2.h5')  # always save your weights after training or during training



if __name__=="__main__":

    # save_bottlebeck_features()
    # train_top_model()

    fine_tune(10)
