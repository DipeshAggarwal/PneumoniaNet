import h5py
import os

class HDF5DatasetWriter():
    
    def __init__(self, dims, output_path, data_key="images", buffer_size=1000, overwrite=True):
        if os.path.exists(output_path) and not overwrite:
            raise ValueError("The supplied path already exists.")
            
        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims, dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],), dtype="int")
        
        self.buffer_size = buffer_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0
        
    def add(self, rows, labels):
        # Add data and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        
        # If buffer is over buffer_size, write to 5 dataset
        if len(self.buffer["data"]) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        i = self.idx + len(self.buffer["data"])
        
        # Write the buffer to disk
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        
        # Reset the buffer and update the index
        self.idx = i
        self.buffer = {"data": [], "labels": []}
        
    def store_class_labels(self, class_labels):
        dt = h5py.special_dtype(vlen=str)
        
        # Create a dataset to store the label names
        label_set = self.db.create_dataset("label_names", (len(class_labels),), dtype=dt)
        label_set[:] = class_labels
        
    def close(self):
        # If there are some entires in the buffer, write them to the dataset
        if (len(self.buffer["data"])) > 0:
            self.flush()
            
        # Close the dataset
        self.db.close()
