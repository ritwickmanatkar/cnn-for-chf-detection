"""
This file contains all the constants utilized by the different processes.
"""
import os

# Base Folder
BASE_PATH = os.getcwd() + "/"

# Base Data Folders
DATA_FOLDER = BASE_PATH + "data/"

# Raw Data Folder
BIDMC_FOLDER_NAME = DATA_FOLDER + "bidmc-congestive-heart-failure-database-1.0.0/"
MITBIH_FOLDER_NAME = DATA_FOLDER + "mit-bih-normal-sinus-rhythm-database-1.0.0/"

# Processed Data Base Folder
PROCESSED_DATA_FOLDER = DATA_FOLDER + "processed_data/"

# Processed Data folders
FIVE_MIN_TIME_SLICES_FOLDER = PROCESSED_DATA_FOLDER + "5min_sections/"
COMBINED_DATA_FOLDER = PROCESSED_DATA_FOLDER + "combined_data/"
EXTRACTED_HEARTBEATS_FOLDER = PROCESSED_DATA_FOLDER + "extracted_heartbeats/"

# DATA PARAMETERS
MITBIH_SAMPLING_FREQUENCY = 128

# ALTERNATE DATA PROCESSING PARAMS
TIME_SERIES_LENGTH = 200
BEATS_SELECTION_DURATION_IN_SECS = 10

# DATA PROCESSING PARAMS
# Found here in Section 2.2: https://sci-hub.usualwant.com/10.1016/j.bspc.2019.101597
PRE_BEAT_WINDOW_SIZE_IN_SECS = 0.235
POST_BEAT_WINDOW_SIE_IN_SECS = 0.390
SELECTION_WINDOW_IN_SECS = 5
N_RANDOM_SAMPLES_IN_SELECTION_WINDOW = 1
SELECTED_ANNOTATIONS = "N"

# MODEL PARAMETERS
N_CLASSES = 2
N_EPOCHS = 10
BATCH_SIZE = 128

# Model Folder
MODEL_FOLDER = BASE_PATH + "model/"
