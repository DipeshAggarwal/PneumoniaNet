import matplotlib
matplotlib.use("Agg")

from core import config
from core.conv import PneumoniaNet
from core.utils.helper import info
from core.callbacks import TrainingMonitor
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import tensorflow_io as tfio
import json

# This function loads the hdf5 data into tf.data compatible form
def load_hdf5(fn):
    images = tfio.IODataset.from_hdf5(fn, "/images")
    labels = tfio.IODataset.from_hdf5(fn, "/labels")
    
    return tf.data.Dataset.zip((images, labels))

# Load the mean values we had saved udring the building database phase
mean = json.loads(open(config.DATASET_MEAN).read())

train_aug = Sequential([
        # Normalise the Mean of the images.
        preprocessing.Normalization(mean=[mean["R"], mean["G"], mean["B"]], variance=0.),
        preprocessing.Rescaling(1./255),
        preprocessing.RandomFlip("horizontal_and_vertical"),
        preprocessing.RandomRotation(0.2),
        preprocessing.RandomZoom(
            height_factor=(-0.05, -0.10),
            width_factor=(-0.05, -0.10)
            )
    ])

test_aug = Sequential([
        preprocessing.Normalization(mean=[mean["R"], mean["G"], mean["B"]], variance=0.),
        preprocessing.Rescaling(1./255)
    ])

val_aug = Sequential([
        preprocessing.Normalization(mean=[mean["R"], mean["G"], mean["B"]], variance=0.),
        preprocessing.Rescaling(1./255)
    ])

# Load the HDF5 dataset
train_dataset = load_hdf5(config.TRAIN_HDF5)
test_dataset = load_hdf5(config.TEST_HDF5)
val_dataset = load_hdf5(config.VAL_HDF5)

train_dataset = (train_dataset
	.cache()
	.batch(config.BATCH_SIZE)
    .map(lambda x, y: (train_aug(x), y), num_parallel_calls=AUTOTUNE)
	.prefetch(AUTOTUNE)
)

test_dataset = (test_dataset
	.cache()
	.batch(config.BATCH_SIZE)
    .map(lambda x, y: (test_aug(x), y), num_parallel_calls=AUTOTUNE)
	.prefetch(AUTOTUNE)
)

val_dataset = (val_dataset
	.cache()
	.batch(config.BATCH_SIZE)
    .map(lambda x, y: (val_aug(x), y), num_parallel_calls=AUTOTUNE)
	.prefetch(AUTOTUNE)
)

info("Initializing model...")
model = PneumoniaNet.build(128, 128, 3, 2)
opt = SGD(learning_rate=config.INIT_LR)

info("Compiling model...")
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

callbacks = [TrainingMonitor(fig_path=config.PLOT_PATH, json_path=config.PLOT_JSON_PATH)]

# train the model
info("Training model...")
H = model.fit(
    train_dataset,
    validation_data=val_dataset,
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    callbacks=callbacks
    )

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(test_dataset)
info("Accuracy: {:.2f}%".format(accuracy * 100))

model.save(config.MODEL_PATH)
