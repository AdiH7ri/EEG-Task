import sys
import os
import mne
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import torch
import argparse

from features import EEGFeatureExtractor
from classicalml import initiate_cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from eegnet import EEGNet, EEGDataset, cross_validate
from config import ExpParams, TrainParams

DATASET_PATH = Path(os.getcwd()) / 'data/eegdata'
SEED = ExpParams.SEED
CV_FOLDS = 5

EPOCH_LENGTH = 2.0  # in seconds
OVERLAP = 0.5       

def extract_epochs(raw : mne.io.Raw)  -> np.ndarray: 
    '''
    Extract fixed-length epochs from raw EEG data.

    Args:
        raw: MNE Raw object

    Returns:
        Numpy array of shape (n_epochs, n_channels, n_timesteps)    

    '''
    epochs = mne.make_fixed_length_epochs(
        raw,
        duration=EPOCH_LENGTH,
        overlap=OVERLAP,
        preload=True,
        verbose=False
    )
    # Convert to numpy: (n_epochs, n_channels, n_timesteps)
    data = epochs.get_data()
    return data

def apply_preprocessing(raw: mne.io.Raw) -> mne.io.Raw:
    '''
    Apply preprocessing steps to raw EEG data.
    
    Steps:
    1. Apply bandpass filter
    2. Apply notch filter
    3. Perform ICA
    4. Standardize data

    Args:
        raw: MNE Raw object 
    
    Returns:
        Preprocessed MNE Raw object
    '''
    # # Applying bandpass filter 
    # modraw.filter(l_freq=0.5, h_freq=45.0, picks='eeg', method='fir', phase='zero')  # FIR because because there is no feedback in the filter, giving results that tend to be stable compared to Infinite Impulse Response (IIR)

    # # Applying notch filter to remove power line noise

    # modraw.notch_filter(freqs=50, picks='eeg')

    # # Perform independent component analysis to remove artifacts (if any)
    raw.set_channel_types({'ECG ECG': 'ecg'})

    # Fit ICA
    ica = mne.preprocessing.ICA(
    n_components=15,
    random_state=42,
    method='fastica', verbose='ERROR'
    )

    ica.fit(raw)

    # Find and remove ECG artifacts
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)
    # ica.plot_scores(scores) 
    
    ica.exclude = ecg_inds
    ica.apply(raw)

    modraw = ica.apply(raw.copy())

    modraw.drop_channels(['ECG ECG'])

    # Access the data as a NumPy array
    data, _ = modraw.get_data(return_times=True)

    # Standardize the data per channel: (x - mean) / std
    data_standardized = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    # # Update the Raw object with the standardized data
    modraw._data = data_standardized

    return modraw

def ml_main(pca_flag) -> None:
    '''
    Main function for machine learning pipeline.
    '''
    mne.set_log_level('WARNING')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    warnings.filterwarnings('ignore')  # suppress all warnings
    
    csv_path = DATASET_PATH / 'subject-info.csv'

    metadata_df = pd.read_csv(csv_path)

    # Prepare data from all subjects
    all_features = []

    for subject_id in metadata_df.Subject.values:

        # eegmat data: 1 = baseline, 2 = arithmetic

        baseline_file_path = DATASET_PATH / f'{subject_id}_1.edf'
        arithmetic_file_path = DATASET_PATH / f'{subject_id}_2.edf'

        baseline_raw = mne.io.read_raw_edf(baseline_file_path, preload=True, verbose=False)
        arithmetic_raw = mne.io.read_raw_edf(arithmetic_file_path, preload=True, verbose=False)
        
        # Preprocess the data
        arithmetic_processed = apply_preprocessing(arithmetic_raw)
        baseline_processed = apply_preprocessing(baseline_raw)
        
        # Extract features
        feature_extractor = EEGFeatureExtractor(sfreq=arithmetic_raw.info['sfreq'])

        baseline_features = feature_extractor.extract_features(baseline_processed)
        baseline_features['label'] = 0  # Relaxed
        baseline_features['subject_id'] = subject_id

        arithmetic_features = feature_extractor.extract_features(arithmetic_processed)
        arithmetic_features['label'] = 1  # Focused
        arithmetic_features['subject_id'] = subject_id
        
        # Combine features for this subject
        subject_features = pd.concat([arithmetic_features, baseline_features], ignore_index=True)
        all_features.append(subject_features)

        print(f'Subject {subject_id} processed!')
            
    all_features_df = pd.concat(all_features, ignore_index=True)

    print(f"Total dataset: {len(all_features_df)} samples, {all_features_df.shape[1]-2} features")
    all_features_df['group'] = all_features_df.apply(lambda row: 'G' if int(metadata_df[ metadata_df['Subject'] == row['subject_id']]['Count quality'].values[0]) == 1 else 'B', axis= 1)

    X = all_features_df[[ x for x in all_features_df.columns if x not in ['label',	'subject_id', 'group']]]
    Y = all_features_df['label']

    if pca_flag == 'N':    
    
        pass

    else:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Choose number of components or variance threshold
        pca = PCA(n_components=0.95, random_state=SEED) 

        # Fit and transform the data
        X_pca = pca.fit_transform(X_scaled)

        print(f"Original feature count: {X.shape[1]}")
        print(f"Reduced feature count: {X_pca.shape[1]}")
        
        X = X_pca

    results_df = initiate_cross_validation(X,Y,SEED,CV_FOLDS)

    results_df.to_csv(DATASET_PATH.parent / 'sml_results.csv', index=False)

    print(f'Results csv saved successfully at {str(DATASET_PATH.parent)} !')

    return None

def dl_main():
    '''
    Main function for deep learning pipeline.
    '''
    mne.set_log_level('WARNING')  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    warnings.filterwarnings('ignore')  # suppress all warnings
    
    csv_path = DATASET_PATH / 'subject-info.csv'

    metadata_df = pd.read_csv(csv_path)

    # Prepare data from all subjects
    all_X, all_y, all_subjects = [], [], []
    for subject_id in metadata_df.Subject.values:

        # eegmat data: 1 = baseline, 2 = arithmetic

        baseline_file_path = DATASET_PATH / f'{subject_id}_1.edf'
        arithmetic_file_path = DATASET_PATH / f'{subject_id}_2.edf'

    
        baseline_raw = mne.io.read_raw_edf(baseline_file_path, preload=True, verbose=False)
        arithmetic_raw = mne.io.read_raw_edf(arithmetic_file_path, preload=True, verbose=False)
        
        # Preprocess
        baseline_processed = apply_preprocessing(baseline_raw)
        arithmetic_processed = apply_preprocessing(arithmetic_raw)

        # Extract epochs as arrays
        baseline_data = extract_epochs(baseline_processed)
        arithmetic_data = extract_epochs(arithmetic_processed)
        
        # Labels: 0 = baseline (relaxed), 1 = arithmetic (focused)
        all_X.append(baseline_data)
        all_y.append(np.zeros(len(baseline_data)))
        all_subjects.extend([subject_id] * len(baseline_data))

        all_X.append(arithmetic_data)
        all_y.append(np.ones(len(arithmetic_data)))
        all_subjects.extend([subject_id] * len(arithmetic_data))

    # Concatenate all subjects’ data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print("Final data shape:", X.shape, "Labels:", y.shape)

    # Initialize dataset
    dataset = EEGDataset(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model constructor (to pass parameters)
    def eegnet_constructor():
        return EEGNet(n_channels=X.shape[1], n_samples=X.shape[2], n_classes=len(np.unique(y)))

    # Run K-fold cross-validation
    cv_summary, test_metrics = cross_validate(
        model_class=eegnet_constructor,
        dataset=dataset,
        save_dir="artifacts_cv",
        n_splits=TrainParams.K,
        n_epochs=TrainParams.TRAINING_EPOCH,
        lr=TrainParams.LEARNING_RATE,
        device=device,
        test_ratio=TrainParams.TEST_RATIO
    )

    print("\nCross-validation Summary:")
    print(cv_summary)
    print("\nFinal Held-out Test Metrics:")
    print(test_metrics)

    return None

if __name__ == "__main__":
    
    # Argument parser for command line inputs
    parser = argparse.ArgumentParser()

    # Option 1 - ML, 2 - DL and PCA - Y/N
    parser.add_argument("Option", type=str, choices=['1','2'], help="1 - ML, 2 - DL")
    parser.add_argument("PCA", type=str, choices=['Y','N'], help="Y/N")
    args = parser.parse_args()

    ch = args.Option
    
    if ch =='1':
        ml_main(args.PCA)

    elif ch =='2':
        dl_main()

    else:
        pass