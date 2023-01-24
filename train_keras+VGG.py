# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,
    help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
    help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
ap.add_argument("-mc", "--modelComplexity", type=int, required=True,
    help="model complexity")
ap.add_argument("-p", "--prep", type=str,
    help="preprocessing. e.g. histogram equalization")
ap.add_argument("-e", "--epochs", type=int, required=True)
ap.add_argument("-s", "--side", type=int, required=True)
ap.add_argument("-b", "--batch", type=int, required=True)
args = vars(ap.parse_args())

# import the necessary packages
if args["modelComplexity"] == 43:
    from src.vgg import vgg
else:
    print("Need to assign model complexity!") 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
import time

# start time 
start_time = time.time()

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = args["batch"]
EPOCHS = args["epochs"]
side = args["side"]  #ori=32

#clip_limit = 2.0
#gridsize = 2

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    # extract the class label from the filename, load the image and
    # resize it to be a fixed 96x96 pixels, ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    #print(label)
    image = cv2.imread(imagePath)

    if args["prep"] == 'clahe':
        # convert image to LAB color model
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # split the image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(image_lab)

        # apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(gridsize,gridsize))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L channel with the original A and B channel
        merged_channels = cv2.merge((cl, a_channel, b_channel))

        # convert iamge from LAB color model back to RGB color model
        image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)


    image = cv2.resize(image, (side, side))

    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
#print(labels.shape)
#Print(labels)
labels = np_utils.to_categorical(labels, 2)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.0, # original shear_range=0.15
    #brightness_range=[0.95, 1.05],
    #zca_whitening=True,
    #featurewise_center=True, featurewise_std_normalization=True,
    #samplewise_center=True, samplewise_std_normalization=True,
    horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = vgg.build(width=side, height=side, depth=3,
    classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# elapsed time
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print("\nElapsed time: {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours),int(minutes),seconds))
