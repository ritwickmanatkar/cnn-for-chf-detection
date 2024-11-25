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

# Processed Data Folder
PROCESSED_DATA_FOLDER = DATA_FOLDER + "processed_data/"

# 5 Min Time Slices folder
FIVE_MIN_TIME_SLICES_FOLDER = PROCESSED_DATA_FOLDER + "5min_sections/"