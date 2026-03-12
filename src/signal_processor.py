import numpy as np 
from scipy.signal import butter, filtfilt

class SignalProcessor:
    """Here we apply bandpass filtering and windowing to the raw 
    EMG signals.
    
    One thing to note, DB! is sampled at 100Hz but Nyquist theorem
    says the max representable frequency is 50Hz so we capped at 45Hz"""

    def __init__(self, fs: int = 100, lowcut: float = 20.0, highcut: float = 45.0, order: int = 4):

        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        self._b, self._a = self._design_filter()

    def _design_filter(self):

        nyquist = self.fs / 2.0
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(self.order, [low, high], btype='band')
        return b, a
    
    def filter(self, emg: np.ndarray) -> np.ndarray: 
        """We are applying zero-phase bandpass filter to all 10 channels."""
        return filtfilt(self._b, self._a, emg, axis=0) 
    
    def segment(self, emg: np.ndarray, labels: np.ndarray, window_ms: int = 200, overlap: float = 0.5):
        window_len = int(self.fs * (window_ms / 1000))
        step = int(window_len * (1 - overlap))

        windows, window_labels = [], []
        for start in range(0, len(emg) - window_len, step):
            end = start + window_len
            window = emg[start:end]

            label_slice = labels[start:end].astype(int)
            majority = np.bincount(label_slice).argmax()

            windows.append(window)
            window_labels.append(majority)

        print(f'Segmentation: {len(windows):,} windows ' 
              f'({window_ms}ms, {int(overlap*100)}% overlap)') 
        
        return np.array(windows), np.array(window_labels)
    
    