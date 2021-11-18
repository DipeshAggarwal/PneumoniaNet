import os

BASE_PATH = "dataset"

TRAIN_DIR = os.path.sep.join([BASE_PATH, "train"])
TEST_DIR = os.path.sep.join([BASE_PATH, "test"])
VAL_DIR = os.path.sep.join([BASE_PATH, "val"])

TRAIN_HDF5 = "hdf5/train.hdf5"
TEST_HDF5 = "hdf5/test.hdf5"
VAL_HDF5 = "hdf5/val.hdf5"

MODEL_PATH = "output/pneumonia.model"
DATASET_MEAN = "output/mean.json"

DIM = (128, 128)

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 64
