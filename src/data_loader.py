import scipy.io as sio
import numpy as np

class EMGDataLoader:
    """We are loading and parsing the .mat files then 
    filtering only the gesture classes we really care about."""

    TARGET_GESTURES = [1, 2, 3, 4, 5, 6] #1-6 are the basic finger flexions
    SAMPLING_RATE = 100 #Hz : DB! recorded at 100Hz

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.emg = None
        self.labels = None
        self.repetitions = None

    def load(self):
        mat = sio.loadmat(self.filepath)

        self.emg = mat['emg'].astype(np.float64)
        self.labels = mat['restimulus'].flatten()
        self.repetitions = mat['repetition'].flatten()

        print(f'Loaded {self.filepath}') 
        print(f'  Samples: {self.emg.shape[0]:,}  Channels: {self.emg.shape[1]}') 
        print(f'  Unique labels found: {sorted(np.unique(self.labels).tolist())}') 
        return self
    
    def filter_gestures(self):
        mask = np.isin(self.labels, self.TARGET_GESTURES)
        self.emg = self.emg[mask]
        self.labels = self.labels[mask]
        self.repetitions = self.repetitions[mask]

        print(f'After filtering to gestures {self.TARGET_GESTURES}:') 
        print(f'  Remaining samples: {self.emg.shape[0]:,}')

        unique, counts = np.unique(self.labels, return_counts=True) 
        for g, c in zip(unique, counts): 
            print(f'  Gesture {g}: {c:,} samples')  
        return self
    
    @staticmethod
    def load_multiple(filepaths: list):
        all_emg, all_labels, all_reps = [], [], []
        for fp in filepaths:
            loader = EMGDataLoader(fp).load().filter_gestures()
            all_emg.append(loader.emg)
            all_labels.append(loader.labels)
            all_reps.append(loader.repetitions)

        emg = np.vstack(all_emg)
        labels = np.concatenate(all_labels)
        repetitions = np.concatenate(all_reps)
        print(f'Combined dataset: {emg.shape[0]:,} samples from {len(filepaths)} subjects') 

        return emg, labels, repetitions