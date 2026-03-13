import numpy as np

class FeatureExtractor:
    """We are extracting 4 time domain features per channel from the EMG window"""

    def extract_window(self, window: np.ndarray) -> np.ndarray:
        """This extracts the feature vector from a single window."""

        features = []
        for ch in range(window.shape[1]):
            x = window[:, ch]
            features.extend([
                self._rms(x), # signal energy
                self._mav(x), # mean activation level
                self._zcr(x), # frequency content proxy
                self._wl(x), # signal complexity
            ])
        return np.array(features)
    
    def extract_all(self, windows: np.ndarray) -> np.ndarray:
        X = np.array([self.extract_window(w) for w in windows])
        print(f'Feature matrix shape: {X.shape} (windows * features)')
        print(f'Feature value range: [{X.min():.4f}, {X.max():.4f}]') 
        return X
    
    def _rms(self, x: np.ndarray) -> float: 
        return float(np.sqrt(np.mean(x ** 2)))
    
    def _mav(self, x: np.ndarray) -> float:
        return float(np.mean(np.abs(x)))
    
    def _zcr(self, x: np.ndarray) -> float:
        signs = np.sign(x)
        signs[signs == 0] = 1
        crossings = np.sum(np.diff(signs) != 0)
        return float(crossings / len(x))
    
    def _wl(self, x: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(x))))
    
    