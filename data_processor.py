"""
This file processes data.
"""
import os
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import wfdb
from wfdb import processing
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import find_peaks

from constants import (
    BIDMC_FOLDER_NAME,
    MITBIH_FOLDER_NAME,
    EXTRACTED_HEARTBEATS_FOLDER,
    MITBIH_SAMPLING_FREQUENCY,
    TIME_SERIES_LENGTH,
    BEATS_SELECTION_DURATION_IN_SECS,
    PRE_BEAT_WINDOW_SIZE_IN_SECS,
    POST_BEAT_WINDOW_SIE_IN_SECS,
    SELECTION_WINDOW_IN_SECS,
    N_RANDOM_SAMPLES_IN_SELECTION_WINDOW
)


@dataclass(frozen=True)
class Dataset:
    MIT_BIH_SINUS_RHYTHM = 'mit-bih-normal-sinus-rhythm-database'
    BIDMC_CONGESTIVE_HEART_FAILURE = 'bidmc_congestive_heart_failure_database'


@dataclass(frozen=True)
class AAMI_EC57_Category:
    """AAMI EC57 Heartbeat Categories."""
    Normal: str = "N"
    Supraventricular: str = "S"
    Ventricular: str = "V"
    Fusion: str = "F"
    Unknown: str = "Q"


def build_file_sets():
    """ This function gives us names of all the files we need to extract for the BIDMC and MITBIH
    datasets.

    :return: [pd.DataFrame containing BIDMC file names, pd.DataFrame containing BIDMC file names]
    """
    df_1 = pd.read_csv(BIDMC_FOLDER_NAME + "RECORDS", header=None)
    df_2 = pd.read_csv(MITBIH_FOLDER_NAME + "RECORDS", header=None)

    return df_1, df_2


def extract_heartbeats_from_ecg_data_file(
        dat_file_path: str,
        dataset: str,
        sampling_frequency_hz: int,
        pre_beat_ann_window_size_in_secs: float,
        post_beat_ann_window_size_in_secs: float,
        selection_window_length_in_secs: int,
        n_random_samples_in_selection_window: int,
        focus_signal: str = "ECG1",
        AAMI_EC57_heartbeat_category: str = AAMI_EC57_Category.Normal
) -> pd.DataFrame:
    """

    :param dat_file_path:
    :param dataset:
    :param sampling_frequency_hz:
    :param pre_beat_ann_window_size_in_secs:
    :param post_beat_ann_window_size_in_secs:
    :param selection_window_length_in_secs:
    :param n_random_samples_in_selection_window:
    :param focus_signal:
    :param AAMI_EC57_heartbeat_category:

    :return:
    """
    # Step 1: Loading Data
    record = wfdb.rdrecord(dat_file_path)
    if dataset == Dataset.BIDMC_CONGESTIVE_HEART_FAILURE:
        ann = wfdb.rdann(dat_file_path, 'ecg')
    else:
        ann = wfdb.rdann(dat_file_path, 'atr')

    # Step 2: Check for the need of down/up sampling
    if record.fs != sampling_frequency_hz:
        record.p_signal, ann = processing.resample_multichan(
            xs=record.p_signal,
            ann=ann,
            fs=record.fs,
            fs_target=sampling_frequency_hz
        )

    # Step 3: Build the Signal Data
    # Rounding is done to avoid unnecessary precision.
    data = np.round(np.array(record.p_signal), 3)
    if data.ndim > 1:
        try:
            selected_idx = record.sig_name.index(focus_signal)
        except ValueError as err:
            print("ECG1 signal not found", err)
            selected_idx = 0
    data = data[:, selected_idx]

    # Step 4: Build the heartbeat annotations
    beats = np.zeros_like(data)
    # Keep only the heartbeats of the expected category.
    beats[ann.sample[np.where(np.array(ann.symbol) == AAMI_EC57_heartbeat_category)]] = 1

    # Step 5: Define the result container
    selected_beats = []

    # Step 6: Split the ECG data into windows for processing
    window_size = sampling_frequency_hz * selection_window_length_in_secs

    # Step 7: Define the heartbeat extraction window
    pre_window_length = int(np.round(pre_beat_ann_window_size_in_secs * sampling_frequency_hz))
    post_window_length = int(np.round(post_beat_ann_window_size_in_secs * sampling_frequency_hz))

    # Logging utils
    count_non_interesting_windows = 0
    count_not_enough_beats_windows = 0
    for window_start in tqdm(range(0, len(data), window_size)):
        # Step 8: Create data subset
        data_window = data[window_start: window_start + window_size]
        beats_window = beats[window_start: window_start + window_size]

        # Step 9: Extract 'n' beats for processing. Boundary cases are also handled
        beat_idxs = np.where(beats_window > 0)[0]
        beat_idxs = beat_idxs[np.where(
            (beat_idxs > pre_window_length) & (beat_idxs < window_size - post_window_length))
        ]

        if len(beat_idxs) == 0:
            count_non_interesting_windows += 1
            continue
        elif len(beat_idxs) < n_random_samples_in_selection_window:
            count_not_enough_beats_windows += 1
            continue
        else:
            # For seed, setting numpy seed should ensure reproducibility.
            selected_idxs = np.random.choice(
                beat_idxs,
                n_random_samples_in_selection_window,
                replace=False
            )

        # Step 9: Slice a selected beat.
        for idx in selected_idxs:
            data_slice = StandardScaler().fit_transform(
                data_window[idx - pre_window_length: idx + post_window_length].reshape(-1, 1)
            ).flatten()

            if len(data_slice) != pre_window_length+ post_window_length:
                print(f"ValueError: Heart beat at idx {idx} has a problem with data length. "
                      f"Length = {len(data)}")
                continue

            selected_beats.append(
                list(data_slice) + [
                    1 if dataset == Dataset.BIDMC_CONGESTIVE_HEART_FAILURE else 0
                ]
            )

    print(f"{len(selected_beats)} heartbeats extracted")
    print(f"{count_non_interesting_windows} windows of data skipped. Did not find any interesting "
          f"heartbeats in them.")
    print(f"{count_not_enough_beats_windows} windows skipped. Did not find enough interesting "
          f"heartbeats to generate requested sample size.")
    df = pd.DataFrame(selected_beats, columns=list(range(len(selected_beats[0]) - 1)) + ['label'])

    return df


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
        fs=record.fs,
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
    ann = wfdb.rdann(dat_file, 'ecg')

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
        processed_bidmc_data[file_name] = extract_heartbeats_from_ecg_data_file(
            dat_file_path=BIDMC_FOLDER_NAME + str(file_name),
            dataset=Dataset.BIDMC_CONGESTIVE_HEART_FAILURE,
            sampling_frequency_hz=MITBIH_SAMPLING_FREQUENCY,
            pre_beat_ann_window_size_in_secs=PRE_BEAT_WINDOW_SIZE_IN_SECS,
            post_beat_ann_window_size_in_secs=POST_BEAT_WINDOW_SIE_IN_SECS,
            selection_window_length_in_secs=SELECTION_WINDOW_IN_SECS,
            n_random_samples_in_selection_window=N_RANDOM_SAMPLES_IN_SELECTION_WINDOW
        )
        cnt += 1

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
        processed_mitbih_data[file_name] = extract_heartbeats_from_ecg_data_file(
            dat_file_path=MITBIH_FOLDER_NAME + str(file_name),
            dataset=Dataset.MIT_BIH_SINUS_RHYTHM,
            sampling_frequency_hz=MITBIH_SAMPLING_FREQUENCY,
            pre_beat_ann_window_size_in_secs=PRE_BEAT_WINDOW_SIZE_IN_SECS,
            post_beat_ann_window_size_in_secs=POST_BEAT_WINDOW_SIE_IN_SECS,
            selection_window_length_in_secs=SELECTION_WINDOW_IN_SECS,
            n_random_samples_in_selection_window=N_RANDOM_SAMPLES_IN_SELECTION_WINDOW
        )
        cnt += 1

    print("Saving MITBIH Files...")
    for file, df in tqdm(processed_mitbih_data.items()):
        df.to_csv(EXTRACTED_HEARTBEATS_FOLDER + f"{file}.csv")

    # Alternate Data Processing Technique
    # # ------------- BIDMC Files -------------
    # print("Processing BIDMC CHF Files...")
    # processed_bidmc_data = {}
    # cnt = 1
    # for i in range(len(bidmc_files)):
    #     print(f"File #{cnt}")
    #     file_name = bidmc_files.iloc[i, 0]
    #     processed_bidmc_data[file_name] = process_bidmc_dat_file(
    #         dat_file=BIDMC_FOLDER_NAME + file_name,
    #         ts_length=TIME_SERIES_LENGTH
    #     )
    #     cnt+=1
    #
    # print("Saving BIDMC Files...")
    # for file, df in tqdm(processed_bidmc_data.items()):
    #     df.to_csv(EXTRACTED_HEARTBEATS_FOLDER + f"{file}.csv")
    #
    # # ------------- MITBIH Files -------------
    # print("Processing MITBIH Normal/Sinus Rhythm Files...")
    # processed_mitbih_data = {}
    # cnt = 1
    # for i in range(len(mitbih_files)):
    #     print(f"File #{cnt}")
    #     file_name = mitbih_files.iloc[i, 0]
    #     processed_mitbih_data[file_name] = process_mitbih_dat_file(
    #         dat_file=MITBIH_FOLDER_NAME + str(file_name),
    #         ts_length=TIME_SERIES_LENGTH
    #     )
    #     cnt+=1
    #
    # print("Saving MITBIH Files...")
    # for file, df in tqdm(processed_mitbih_data.items()):
    #     df.to_csv(EXTRACTED_HEARTBEATS_FOLDER + f"{file}.csv")
