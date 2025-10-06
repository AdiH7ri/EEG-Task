from dataclasses import dataclass
from pathlib import Path

# Configuration parameters for the experiments

@dataclass
class ExpParams:
    SEED = 7

@dataclass
class TrainParams:
    K = 10
    TRAINING_EPOCH = 150
    TRAIN_CUDA = True
    IN_CHANNELS = 1
    OUT_CHANNELS = 3
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 128
    TEST_RATIO= 0.2
