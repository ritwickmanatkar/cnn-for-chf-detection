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
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks


from constants import (
    BIDMC_FOLDER_NAME,
    MITBIH_FOLDER_NAME,
    EXTRACTED_HEARTBEATS_FOLDER,
    MITBIH_SAMPLING_FREQUENCY,
    TIME_SERIES_LENGTH,
    BEATS_SELECTION_DURATION_IN_SECS
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

    Heartbeat Extraction using R-Peak identification method is from this paper:
    https://arxiv.org/pdf/1805.00794
    """
    record = wfdb.rdrecord(dat_file)
    ann = wfdb.rdann(dat_file, 'ecg')

    # Step 1: Downsampling to 128 Hz
    record.p_signal, ann = processing.resample_multichan(
        xs=record.p_signal,
        ann=ann,
        fs=record.fs ,
        fs_target=MITBIH_SAMPLING_FREQUENCY
    )

    # Step 2: Extracting the Time Series ECG1 data
    data = np.round(np.array(record.p_signal), 3)
    if data.ndim > 1:
        try:
            selected_idx = record.sig_name.index('ECG1')
        except ValueError as err:
            print("ECG1 signal not found", err)
            selected_idx = 0
    data = data[:, selected_idx]

    # TODO: I dont think this is useful but keeping it for now.
    if len(data) < ts_length:
        raise ValueError(
            f"'ts_length'= {ts_length} is longer than the length of the data ({len(data)})")

    # Step 3: Preparing the Annotations for the data selection.
    labels = np.zeros_like(data)
    labels[ann.sample] = 1

    # Step 4: Process the R-peaks of the ECG data
    selected_parts = []

    # Step 4.1: Split the file into 10 sec windows
    window_size = MITBIH_SAMPLING_FREQUENCY * BEATS_SELECTION_DURATION_IN_SECS
    for window_start in tqdm(range(0, len(data), window_size)):
        data_window = data[window_start: window_start + window_size]
        label_window = labels[window_start: window_start + window_size]

        # Step 4.2: MinMax Scale the data window
        transformed_data_window = MinMaxScaler().fit_transform(data_window.reshape(-1, 1)).flatten()

        # Step 4.3: Find the set of valid R-peaks from the window.
        r_peak_candidates, _ = find_peaks(x=transformed_data_window, height=0.9, distance=30)

        if len(r_peak_candidates) < 2:
            continue

        # Step 4.4: Calculate R-R intervals and median heartbeat period T
        T = np.median(np.subtract(r_peak_candidates[1:], r_peak_candidates[:-1]))

        # Step 4.5: Calculate the extraction length for this window.
        extraction_length = min(int(1.2 * T), ts_length)

        for r_peak in r_peak_candidates:
            if r_peak + extraction_length < window_size:
                selected_parts.append(
                    # ECG Signal of set length
                    list(
                        np.pad(
                            transformed_data_window[r_peak: r_peak + extraction_length],
                            (0, ts_length - extraction_length),
                            mode='constant'
                        )[:ts_length]
                    ) + [
                        # Label for the extracted beat
                        1 if np.any(label_window[r_peak: r_peak + extraction_length]) else 0
                    ] + [
                        # Annotations for the extracted beat.
                        np.where(label_window[r_peak: r_peak + extraction_length] > 0)[0]
                    ]
                )

    print(f"{len(selected_parts)} heartbeats extracted")
    df = pd.DataFrame(selected_parts, columns=list(range(ts_length)) + ['label', 'annotations'])

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

    # Step 4: Process the R-peaks of the ECG data
    selected_parts = []

    # Step 4.1: Split the file into 10 sec windows
    window_size = MITBIH_SAMPLING_FREQUENCY * BEATS_SELECTION_DURATION_IN_SECS
    for window_start in tqdm(range(0, len(data), window_size)):
        data_window = data[window_start: window_start + window_size]

        # Step 4.2: MinMax Scale the data window
        transformed_data_window = MinMaxScaler().fit_transform(data_window.reshape(-1, 1)).flatten()

        # Step 4.3: Find the set of valid R-peaks from the window.
        r_peak_candidates, _ = find_peaks(x=transformed_data_window, height=0.9, distance=30)

        if len(r_peak_candidates) < 2:
            continue

        # Step 4.4: Calculate R-R intervals and median heartbeat period T
        T = np.median(np.subtract(r_peak_candidates[1:], r_peak_candidates[:-1]))

        # Step 4.5: Calculate the extraction length for this window.
        extraction_length = min(int(1.2 * T), ts_length)

        for r_peak in r_peak_candidates:
            if r_peak + extraction_length < window_size:
                selected_parts.append(
                    # ECG Signal of set length
                    list(
                        np.pad(
                            transformed_data_window[r_peak: r_peak + extraction_length],
                            (0, ts_length - extraction_length),
                            mode='constant'
                        )[:ts_length]
                    ) + [0] + [[]]
                )

    print(f"{len(selected_parts)} heartbeats extracted")
    df = pd.DataFrame(selected_parts, columns=list(range(ts_length)) + ['label', 'annotations'])

    return df


if __name__ == '__main__':
    bidmc_files, mitbih_files = build_file_sets()

    # ------------- BIDMC Files -------------
    print("Processing BIDMC CHF Files...")
    processed_bidmc_data = {}
    cnt = 1
    for i in range(len(bidmc_files)):
        print(f"File #{cnt}")
        file_name = bidmc_files.iloc[i, 0]
        processed_bidmc_data[file_name] = process_bidmc_dat_file(
            dat_file=BIDMC_FOLDER_NAME + file_name,
            ts_length=TIME_SERIES_LENGTH
        )
        cnt+=1

    print("Saving BIDMC Files...")
    for file, df in tqdm(processed_bidmc_data.items()):
        df.to_csv(EXTRACTED_HEARTBEATS_FOLDER + f"{file}.csv")

    # ------------- MITBIH Files -------------
    print("Processing MITBIH Normal/Sinus Rhythm Files...")
    processed_mitbih_data = {}
    cnt = 1
    for i in range(len(mitbih_files)):
        print(f"File #{cnt}")
        file_name = mitbih_files.iloc[i, 0]
        processed_mitbih_data[file_name] = process_mitbih_dat_file(
            dat_file=MITBIH_FOLDER_NAME + str(file_name),
            ts_length=TIME_SERIES_LENGTH
        )
        cnt+=1

    print("Saving MITBIH Files...")
    for file, df in tqdm(processed_mitbih_data.items()):
        df.to_csv(EXTRACTED_HEARTBEATS_FOLDER + f"{file}.csv")
