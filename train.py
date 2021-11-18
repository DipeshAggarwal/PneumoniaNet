import matplotlib
matplotlib.use("Agg")

from core.conv import PneumoniaNet
from core.utils.helper import info
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.data import AUTOTUNE
from core import config

import tensorflow as tf
import tensorflow_io as tfio

# This function loads the hdf5 data into tf.data compatible form
def load_hdf5(fn):
    images = tfio.IODataset.from_hdf5(fn, "/images")
    labels = tfio.IODataset.from_hdf5(fn, "/labels")
    
    return tf.data.Dataset.zip((images, labels))

# Load the HDF5 dataset
train_dataset = load_hdf5(config.TRAIN_HDF5)
test_dataset = load_hdf5(config.TEST_HDF5)
val_dataset = load_hdf5(config.VAL_HDF5)

train_dataset = (train_dataset
	.cache()
	.batch(64)
	.prefetch(AUTOTUNE)
)

test_dataset = (test_dataset
	.cache()
	.batch(64)
	.prefetch(AUTOTUNE)
)

val_dataset = (val_dataset
	.cache()
	.batch(64)
	.prefetch(AUTOTUNE)
)

info("Initializing model...")
model = PneumoniaNet.build(128, 128, 3, 2)

info("Compiling model...")
model.compile(loss="sparse_categorical_crossentropy",
	optimizer="sgd", metrics=["accuracy"])

# train the model
info("Training model...")
H = model.fit(
	train_dataset,
	validation_data=val_dataset,
    batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_dataset)
info("Accuracy: {:.2f}%".format(accuracy * 100))
