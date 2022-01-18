import numpy as np
import tensorflow.keras as keras

class KerasSequence(keras.utils.Sequence):
    def __init__(self, entries:list, batch_size:int):
        self.entries = entries
        self.batch_size = batch_size
    
    def __len__(self):
        return (np.ceil(len(self.entries) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx):
        batch_x = [entry.x() for entry in self.entries[idx * self.batch_size : (idx+1) * self.batch_size]]
        batch_y = [entry.y() for entry in self.entries[idx * self.batch_size : (idx+1) * self.batch_size]]
        
        return np.array(batch_x), np.array(batch_y)
