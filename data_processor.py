"""
This file processes data.
"""
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
from tqdm import tqdm


from constants import (
    BIDMC_FOLDER_NAME,
    MITBIH_FOLDER_NAME,
    FIVE_MIN_TIME_SLICES_FOLDER,
    MITBIH_SAMPLING_FREQUENCY,
    TIME_SERIES_LENGTH
)





def build_file_sets():
    """ This function gives us names of all the files we need to extract for the BIDMC and MITBIH
    datasets.

    :return: [pd.DataFrame containing BIDMC file names, pd.DataFrame containing BIDMC file names]
    """
    df_1 = pd.read_csv(BIDMC_FOLDER_NAME+"RECORDS", header=None)
    df_2 = pd.read_csv(MITBIH_FOLDER_NAME+"RECORDS", header=None)

    return df_1, df_2


def process_bidmc_dat_file(dat_file: str, ts_length: int):
    """ This function processes the BIDMC Congestive Heart Failure Data.
    This data contains ECG signals of patients and notations made by an automated detector.
    The annotations will be used to denote that a segment contains CHF marker.

    NOTE: The param 'dat_file' should not contain the file extension as we use this structure
    to extract the annotation file.
    """
    record = wfdb.rdrecord(dat_file)
    ann = wfdb.rdann(dat_file, 'ecg')

    # Downsampling to 128 Hz
    record.p_signal, ann = processing.resample_multichan(
        xs=record.p_signal,
        ann=ann,
        fs=record.fs ,
        fs_target=MITBIH_SAMPLING_FREQUENCY
    )

    # Extracting the Time Series ECG data
    data = np.array(record.p_signal)
    if data.ndim > 1:
        try:
            selected_idx = record.sig_name.index('ECG1')
        except ValueError as err:
            print("ECG1 signal not found", err)
            selected_idx = 0

    data = data[:, selected_idx]

    if len(data) < ts_length:
        raise ValueError(
            f"'ts_length'= {ts_length} is longer than the length of the data ({len(data)})")

    # Extracting the Annotations for the data.
    labels = np.zeros_like(data)
    labels[ann.sample] = 1

    # Utils for iteration
    slices = []

    for idx in tqdm(range(ts_length, len(data), ts_length)):
        current_slice_of_data = data[idx - ts_length: idx]
        current_label = 1 if np.any(labels[idx - ts_length: idx]) else 0

        slices.append(list(current_slice_of_data) + [current_label])

    # Generated the last slice which was potentially cut off.
    if len(data) % ts_length != 0:
        current_slice_of_data = data[len(data) - ts_length:]
        current_label = 1 if np.any(labels[len(data) - ts_length:]) else 0

        slices.append(list(current_slice_of_data) + [current_label])

    df = pd.DataFrame(slices, columns=list(range(ts_length)) + ['label'])

    return df


def process_mitbih_dat_file(dat_file: str, ts_length: int):
    """ This function processes the MITBIH Normal/Sinus Rhythm Data.
    This data contains ECG signals of patients. There are no labels for this.

    NOTE: The param 'dat_file' should not contain the file extension. This is keeping with the
    other functions format.
    """
    record = wfdb.rdrecord(dat_file)

    # Extracting the Time Series ECG data
    data = np.array(record.p_signal)
    if data.ndim > 1:
        try:
            selected_idx = record.sig_name.index('ECG1')
        except ValueError as err:
            print("ECG1 signal not found", err)
            selected_idx = 0
    data = data[:, selected_idx]

    if len(data) < ts_length:
        raise ValueError(
            f"'ts_length'= {ts_length} is longer than the length of the data ({len(data)})")

    # Utils for iteration
    slices = []

    for idx in tqdm(range(ts_length, len(data), ts_length)):
        current_slice_of_data = data[idx - ts_length: idx]
        slices.append(list(current_slice_of_data) + [0])

    # Generated the last slice which was potentially cut off.
    if len(data) % ts_length != 0:
        current_slice_of_data = data[len(data) - ts_length:]
        slices.append(list(current_slice_of_data) + [0])

    df = pd.DataFrame(slices, columns=list(range(ts_length)) + ['label'])
    return df


if __name__ == '__main__':
    bidmc_files, mitbih_files = build_file_sets()

    # ------------- BIDMC Files -------------
    print("Processing BIDMC CHF Files...")
    processed_bidmc_data = {}
    for i in tqdm(range(len(bidmc_files))):
        file_name = bidmc_files.iloc[i, 0]
        processed_bidmc_data[file_name] = process_bidmc_dat_file(
            dat_file=BIDMC_FOLDER_NAME + file_name,
            ts_length=TIME_SERIES_LENGTH
        )

    print("Saving BIDMC Files...")
    for file, df in tqdm(processed_bidmc_data.items()):
        df.to_csv(FIVE_MIN_TIME_SLICES_FOLDER + f"{file}.csv")

    # ------------- MITBIH Files -------------
    print("Processing MITBIH Normal/Sinus Rhythm Files...")
    processed_mitbih_data = {}
    for i in tqdm(range(len(mitbih_files))):
        file_name = mitbih_files.iloc[i, 0]
        processed_mitbih_data[file_name] = process_mitbih_dat_file(
            dat_file=MITBIH_FOLDER_NAME + str(file_name),
            ts_length=TIME_SERIES_LENGTH
        )

    print("Saving MITBIH Files...")
    for file, df in tqdm(processed_mitbih_data.items()):
        df.to_csv(FIVE_MIN_TIME_SLICES_FOLDER + f"{file}.csv")
