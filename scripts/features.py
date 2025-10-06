import numpy as np
import pandas as pd
import mne
from scipy import signal
from typing import Dict, List, Tuple, Union
import pywt


class EEGFeatureExtractor:
    """Extracts various brainwave features from preprocessed EEG data."""
    
    # Define frequency bands of interest
    BRAINWAVE_BANDS = {
        'delta': (0.5, 4),    # Sleep, deep relaxation
        'theta': (4, 8),      # Drowsiness, meditation, creativity
        'alpha': (8, 13),     # Relaxation, calmness, reflection
        'beta': (13, 30),     # Active thinking, focus, alertness
        'gamma': (30, 45)     # Higher cognitive processing, perception
    }
    
    def __init__(self, sfreq: float = 500.0):
        """
        Initialize the feature extractor.
        
        Args:
            sfreq: Sampling frequency of the EEG data (Hz)
        """
        self.sfreq = sfreq
    
    def extract_features(self, 
                         data: Union[mne.io.Raw, mne.Epochs, np.ndarray]):
        """
        Extract features from EEG data.
        
        Args:
            data: EEG data (MNE Raw/Epochs object or numpy array)
            include_bands: List of frequency bands to include 
                           (default: all bands)
            include_connectivity: Whether to include connectivity features
            include_complexity: Whether to include complexity measures
            
        Returns:
            DataFrame with extracted features
        """
        # Determine which bands to extract
        signals = data.get_data()
        bands = list(self.BRAINWAVE_BANDS.keys())
        
        # Initialize features dictionary
        features = {}
        
        # Extract band power features
        band_powers = self._extract_band_powers(signals, bands)
        features.update(band_powers)
        
        # Extract band power ratios
        power_ratios = self._extract_power_ratios(band_powers)
        features.update(power_ratios)

        # Extract hjorth parameters
        hjorth_params = self._extract_hjorth_parameters(signals)
        features.update(hjorth_params)

        # Extract wavelet features
        wvlt_feats = self.extract_wavelet_features(signals)
        features.update(wvlt_feats)
        
        # Convert to DataFrame - for scalar values, we need to create a DataFrame with a single row
        df = pd.DataFrame(features, index=[0])
        
        return df
    
    def _extract_band_powers(self, signals: np.ndarray, 
                            include_bands: List[str]) -> Dict:
        """
        Extract power in different frequency bands.
        
        Args:
            signals: EEG signals with shape (channels, samples)
            include_bands: List of frequency bands to include
            
        Returns:
            Dictionary with band power features
        """
        features = {}
        n_channels = signals.shape[0]
        
        for band_name in include_bands:
            if band_name not in self.BRAINWAVE_BANDS:
                continue
                
            fmin, fmax = self.BRAINWAVE_BANDS[band_name]
            
            # Calculate power spectral density
            for ch_idx in range(n_channels):
                # Compute power spectrum
                f, psd = signal.welch(signals[ch_idx], fs=self.sfreq, 
                                     nperseg=min(256, signals.shape[1]//2))
                
                # Find indices corresponding to the frequency band
                idx_band = np.logical_and(f >= fmin, f <= fmax)
                
                # Calculate average power in the band
                power = np.mean(psd[idx_band])
                
                # Store in features dictionary
                feature_name = f"{band_name}_power_ch{ch_idx+1}"
                features[feature_name] = power
                
            # Also compute average across channels
            features[f"{band_name}_power_avg"] = np.mean(
                [features[f"{band_name}_power_ch{ch_idx+1}"] for ch_idx in range(n_channels)]
            )
        
        return features
    
    def _extract_power_ratios(self, band_powers: Dict) -> Dict:
        """
        Calculate ratios between different frequency bands.
        
        Args:
            band_powers: Dictionary with band power features
            
        Returns:
            Dictionary with band power ratio features
        """
        features = {}
        
        # Define interesting ratios
        ratios = [
            ('theta', 'beta'),    # Relaxation & focus
            ('alpha', 'beta'),    # Relaxation & active thinking
            ('theta', 'alpha'),   # Meditation & relaxed alertness
            ('delta', 'beta')     # Deep relaxation & active thinking
        ]
        
        # Calculate average band powers across channels
        for numerator, denominator in ratios:
            num_key = f"{numerator}_power_avg"
            denom_key = f"{denominator}_power_avg"
            
            if num_key in band_powers and denom_key in band_powers:
                ratio_name = f"{numerator}_{denominator}_ratio"
                
                # Avoid division by zero
                if band_powers[denom_key] > 0:
                    ratio = band_powers[num_key] / band_powers[denom_key]
                else:
                    ratio = 0
                    
                features[ratio_name] = ratio
        
        return features

    def _extract_hjorth_parameters(self, signals: np.ndarray) -> Dict:
        """
        Extract Hjorth parameters (Activity, Mobility, Complexity) from EEG signals.
        
        Args:
            signals: EEG signals with shape (channels, samples)
        
        Returns:
            Dictionary with Hjorth features
        """
        features = {}
        n_channels = signals.shape[0]

        for ch_idx in range(n_channels):

            # Compute first and second derivatives
            signal = signals[ch_idx]
            first_deriv = np.diff(signal)
            second_deriv = np.diff(first_deriv)

            # Compute Hjorth parameters
            activity = np.var(signal)
            mobility = np.sqrt(np.var(first_deriv) / activity) if activity > 0 else 0
            complexity = (np.sqrt(np.var(second_deriv) / np.var(first_deriv))
                        / mobility) if mobility > 0 and np.var(first_deriv) > 0 else 0

            features[f'hjorth_activity_ch{ch_idx+1}'] = activity
            features[f'hjorth_mobility_ch{ch_idx+1}'] = mobility
            features[f'hjorth_complexity_ch{ch_idx+1}'] = complexity

        # Average across channels
        features['hjorth_activity_avg'] = np.mean([features[f'hjorth_activity_ch{i+1}'] for i in range(n_channels)])
        features['hjorth_mobility_avg'] = np.mean([features[f'hjorth_mobility_ch{i+1}'] for i in range(n_channels)])
        features['hjorth_complexity_avg'] = np.mean([features[f'hjorth_complexity_ch{i+1}'] for i in range(n_channels)])

        return features
    
    def extract_wavelet_features(self, signals: np.ndarray, 
                                wavelet: str = 'db4',
                                level: int = 5) -> Dict:
        """
        Extract wavelet-based features from EEG signals.
        
        Args:
            signals: EEG signals with shape (channels, samples)
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary with wavelet features
        """
        features = {}
        n_channels = signals.shape[0]
        
        for ch_idx in range(n_channels):
            signal = signals[ch_idx]
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            
            # Extract features from each coefficient level
            for i, coef in enumerate(coeffs):
                # Calculate energy
                energy = np.sum(coef**2) / len(coef)
                features[f"wavelet_energy_ch{ch_idx+1}_level{i}"] = energy
                
                # Calculate entropy
                if np.any(coef):
                    p = coef**2 / np.sum(coef**2)
                    p = p[p > 0]  # Avoid log(0)
                    entropy = -np.sum(p * np.log2(p))
                else:
                    entropy = 0
                features[f"wavelet_entropy_ch{ch_idx+1}_level{i}"] = entropy
        
        return features