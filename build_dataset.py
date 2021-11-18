from core import config
from core.utils.helper import info
from core.io import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import numpy as np
import progressbar
import random
import json
import cv2
import os

train_paths = list(paths.list_images(config.TRAIN_DIR))
test_paths = list(paths.list_images(config.TEST_DIR))
val_paths = list(paths.list_images(config.VAL_DIR))

random.shuffle(train_paths)
random.shuffle(test_paths)
random.shuffle(val_paths)

train_labels = [p.split(os.path.sep)[-2] for p in train_paths]
test_labels = [p.split(os.path.sep)[-2] for p in test_paths]
val_labels = [p.split(os.path.sep)[-2] for p in val_paths]

le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
test_labels = le.fit_transform(test_labels)
val_labels = le.transform(val_labels)

(R, G, B) = ([], [], [])

dataset = [
        ["train", train_paths, train_labels, config.TRAIN_HDF5],
        ["test", test_paths, test_labels, config.TEST_HDF5],
        ["val", val_paths, val_labels, config.VAL_HDF5],
    ]

for name, image_paths, labels, output_path in dataset:
    info("Building {}...".format(output_path))
    writer = HDF5DatasetWriter((len(image_paths), 128, 128, 3), output_path)
    writer.store_class_labels(le.classes_)
    
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(path)
        image = cv2.resize(image, config.DIM, interpolation=cv2.INTER_AREA)
        
        if name == "train":
            b, g, r = cv2.split(image)
            R.append(r)
            G.append(g)
            B.append(b)
            
        writer.add([image], [label])
        pbar.update(i)
        
    pbar.finish()
    writer.close()

info("Serializing Means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
