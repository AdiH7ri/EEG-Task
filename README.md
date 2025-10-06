# EEG-Task

Scripts used in this project explore EEG-based cognitive state classification. We preprocess EEG signals, extract features (PSD, Hjorth, Wavelet, band ratios), and experiment with machine learning models and deep learning architectures (e.g., EEGNet) to distinguish between relaxed and focused mental states.
## Dataset

The dataset can be found here -> [EEGMat](https://physionet.org/content/eegmat/1.0.0/)

## Installing the dependencies

You can set up the environment using either `pip` or Anaconda.

### Install necessary packages using `pip`

```bash
pip install -r requirements.txt
```

### Set-up a new environment (Anaconda)

To create a new environment with all required packages:

```bash
conda env create --name <your-envname> --file=environment.yml
```

## Folder Structure

The repository expects EEG data and metadata to be organized as follows:
```bash
ROOT/DATASET_PATH/
├── subject-info.csv # Metadata for subjects
├── <subject_id>_1.edf # Baseline EEG recordings
├── <subject_id>_2.edf # Arithmetic/focused EEG recordings
...
```

- `subject-info.csv` contains subject IDs and quality counts.  
- EEG recordings should be in **EDF format**.  
- Processed outputs and intermediate results are saved in user-specified directories during execution.

## Scripts

### Feature Extraction

Edit `scripts/config.py` to change experiment parameters.

### Model Training & Cross-Validation

Run `scripts/main.py` to train models using either classical machine learning algorithms (ET, RF, SVM, MLP) or deep learning (EEGNet). The script supports **Stratified K-Fold cross-validation** with an optional held-out test set.

**Arguments / Parameters:**
- `--option`: 1 for Machine learning pipeline, 2 for Deep learning pipeline
- `--pca`: to run PCA for feature reduction (Y/N)

**Outputs include:**
- Per-fold training/validation metrics  
- Best model weights  
- Cross-validation summary CSV  
- Held-out test metrics JSON 
