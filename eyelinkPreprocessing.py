# Import packages
import csv
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import os
import yaml
import argparse
from pathlib import Path


def load_config(config_path='preprocessingConfig.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
    
def read_eyelink_data(path):
    """ 
    Read in the file and store the data in a dataframe. The file has the following structure:
    * header indicated with '**' at the beginning of each line
    * messages containing information about callibration/validation etc. indicated with 'MSG' at the beginning of each line
    
    This is followed by:
    START	10350638 	LEFT	SAMPLES	EVENT
    PRESCALER	1
    VPRESCALER	1
    PUPIL	AREA
    EVENTS	GAZE	LEFT	RATE	 1000.00	TRACKING	CR	FILTER	2
    SAMPLES	GAZE	LEFT	RATE	 1000.00	TRACKING	CR	FILTER	2
    
    This is followed by the actual data containing the following data types:
    * Samples: [TIMESTAMP]\t [X-Coords]\t [Y-Coords]\t [Diameter]
    * Messages: MSG [TIMESTAMP]\t [MESSAGE], e.g.
        - Trial #1: type = rivalry, report = 1, image1 = face, angle1 = 0°
        - Fix point 1 position: 3
        - Fix point 2 position: 6
        - Current percept: face
    * Events: 
        - SFIX (Start Fixation): SFIX [EYE (L/R)]\t [START TIME]\t 
        - EFIX (End Fixation): EFIX [EYE (L/R)]\t [START TIME]\t [END TIME]\t [DURATION]\t [AVG X]\t [AVG Y]\t [AVG PUPIL]\t
        - SSACC (Start Saccade): SSACC [EYE (L/R)]\t [START TIME]\t 
        - ESACC (End Saccade): ESACC [EYE (L/R)]\t [START TIME]\t [END TIME]\t [DURATION]\t [START X]\t [START Y]\t [END X]\t [END Y]\t [AMP]\t [PEAK VEL]\t
        - SBLINK (Start Blink): SBLINK [EYE (L/R)]\t [START TIME]\t 
        - EBLINK (End Blink): SBLINK [EYE (L/R)]\t [START TIME]\t [DURATION]

    Input: 
        path: str, path to the file
        
    Output:
        df: pd.DataFrame, dataframe containing the data
    """

    # Initialize dataframe
    df = pd.DataFrame()

    # Initialize lists to store the relevant data 
    timestamps = []
    x_coords = []
    y_coords = []
    diameters = []
    saccade_timestamps = []
    fixation_timestamps = []
    blink_timestamps = []
    trials = []
    types = []
    images1 = []
    fixpoint1_positions = []
    fixpoint2_positions = []
    percepts = []

    # Iterate over the file, extract the data, collect it into the lists and adjust the values if necessary
    try:
        with open(path) as f:
           file = csv.reader(f, delimiter='\t')
           start = False
           trial = None
           type = None
           image1 = None
           fixpoint1 = None
           fixpoint2 = None
           percept = None
           for i, row in enumerate(file):
                # Skip header (everything until message includes 'SYNCTIME')
                if not start: 
                    # Extract gaze coordinates
                    if any('GAZE_COORDS' in item for item in row):
                        gaze_coords = row[1].split(' ')[2:]
                        gaze_coords = [float(coord) for coord in gaze_coords]
                    # Extract sampling rate (number that follow 'RATE')
                    elif any('RATE' in item for item in row):
                        sampling_rate = float(row[4])
                    elif any('SYNCTIME' in item for item in row):
                        start = True
                    continue
                # Extract fixations, saccades, blinks, trials, messages and events
                if any('SFIX' in item for item in row): continue
                elif any('EFIX' in item for item in row):
                    fixation_timestamps.append([row[0].split(' ')[4], row[1]])
                    continue
                elif any('SSACC' in item for item in row): continue
                elif any('ESACC' in item for item in row):
                    saccade_timestamps.append([row[0].split(' ')[3], row[1]])
                    continue
                elif any('SBLINK' in item for item in row): continue
                elif any('EBLINK' in item for item in row):
                    blink_timestamps.append([row[0].split(' ')[2], row[1]])
                    continue
                # Extract trial information
                elif any('Trial' in item for item in row):
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    trial = re.search(r'Trial #(\d+)',  message).group(1) # Extract trial number
                    type = re.search(r'type = (\w+)', message).group(0).split(' = ')[1] # Extract trial type
                    image1 = re.search(r'image1 = (\w+)', message).group(0).split(' = ')[1] # Extract image1
                    percept = image1 # Set initial percept to image1
                    continue
                elif any('End of trial' in item for item in row):
                    trial = None
                    type = None
                    fixpoint1 = None
                    fixpoint2 = None
                    percept = None
                    continue
                elif any('Fix point 1' in item for item in row): 
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    fixpoint1 = re.search(r'position: (\d+)',  message).group(1) # Extract fix point position
                    continue
                elif any('Fix point 2' in item for item in row):
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    fixpoint2 = re.search(r'position: (\d+)',  message).group(1)
                    continue
                elif any('Current' in item for item in row): # extract word after ':' in 'Current percept: mixed'
                    message = ''.join(row[1:]).strip() # Join string and remove 'MSG'
                    percept = re.search(r'percept: (\w+)',  message).group(1)
                    continue
                elif any('MSG' in item for item in row):
                    continue
                # Stop at the end of the file
                elif any('END' in item for item in row):
                    break  
                # Extract timestamp and convert to float and divide by 1000 to convert from ms to s
                timestamp = row[0].strip()
                timestamp = float(timestamp)/sampling_rate
                timestamps.append(timestamp)
                # Extract x and y coordinates and convert to float. Set to NaN if data is missing
                x = row[1].strip()
                x = np.nan if x == '.' else float(x)
                x_coords.append(x)
                y = row[2].strip()
                y = np.nan if y == '.' else float(y)
                y_coords.append(y)  
                # Extract diameter and convert to float
                diameter = float(row[3].strip())  
                diameters.append(diameter)
                # Append trial information to lists
                trials.append(trial)
                types.append(type)
                images1.append(image1)
                fixpoint1_positions.append(fixpoint1)
                fixpoint2_positions.append(fixpoint2)
                percepts.append(percept)


        # For fixations, saccades and blinks create a list as long as the timestamps list with zeros
        fixations = [0] * len(timestamps)
        saccades = [0] * len(timestamps)
        blinks = [0] * len(timestamps)

        # Extract the start and end times of fixations, saccades and blinks and set the values in the respective lists to 1
        for i in range(len(fixation_timestamps)):
            start = float(fixation_timestamps[i][0])/sampling_rate
            end = float(fixation_timestamps[i][1])/sampling_rate
            for j in range(len(timestamps)):
                if timestamps[j] >= start and timestamps[j] <= end:
                    fixations[j] = 1
        for i in range(len(saccade_timestamps)):
            start = float(saccade_timestamps[i][0])/sampling_rate
            end = float(saccade_timestamps[i][1])/sampling_rate
            for j in range(len(timestamps)):
                if timestamps[j] >= start and timestamps[j] <= end:
                    saccades[j] = 1
        for i in range(len(blink_timestamps)):
            start = float(blink_timestamps[i][0])/sampling_rate
            end = float(blink_timestamps[i][1])/sampling_rate
            for j in range(len(timestamps)):
                if timestamps[j] >= start and timestamps[j] <= end:
                    blinks[j] = 1

        # Summarize data
        n_saccades = len(saccade_timestamps)
        n_fixations = len(fixation_timestamps)
        n_blinks = len(blink_timestamps)

        # Create a dataframe from the lists
        df = pd.DataFrame(list(zip(timestamps, x_coords, y_coords, diameters, fixations, saccades, blinks, trials, types, images1, fixpoint1_positions, fixpoint2_positions, percepts)), 
                   columns =['Timestamp', 'X', 'Y', 'Diameter', 'Fixation', 'Saccade', 'Blink', 'Trial', 'Type', 'Image1', 'Fixpoint1', 'Fixpoint2', 'Percept'])
        
        return df, gaze_coords, sampling_rate, n_saccades, n_fixations, n_blinks
    except Exception as e:
        raise Warning(f'Could not read ' + str(path) + ' properly! Error: {e}')
    
    
def interpolate_blinks(df, blink_padding, saccade_padding, diameter_threshold, plot, sampling_rate, screen_resolution, x_offset, y_offset, trial):
    """
    Interpolate blinks in the data. The function performs the following steps:
    1. Search for undetected blinks and set them to NaN
    2. Merge two blinks into 1 if they are < xy ms togehter (coalesce blinks)
    3. Pad the blinks
    4. Set saccades to NaN and pad them
    5. Find values beyond the screen and set them to NaN
    6. Detect extreme values for diameter and set to NaN (median +/- 3 std)
    7. Interpolate NaN values

    Input:
        df_preprocessed: pd.DataFrame, dataframe containing the data
        blink_padding: float, padding value for blinks
        diameter_threshold: int, threshold for the diameter of the pupil
        saccade_padding: float, padding value for saccades
        plot: bool, plot the data before and after interpolation
        sampling_rate: int, sampling rate of the data
        screen_resolution: list, screen resolution in pixels
        x_offset: int, x offset of the screen due to fusion alignment
        y_offset: int, y offset of the screen due to fusion alignment
        trial: str, trial number
    
    Output:
        df_interpolated: pd.DataFrame, dataframe containing the data after interpolation
    """
    try: 
        # Check whether there are blinks in the data
        n_blinks = df['Blink'].sum()
        if n_blinks == 0:
            print(f'No blinks detected in trial {trial}.')

        if plot:
            # Plot the data before interpolation. Diameter and gaze separately.
            fig, axs = plt.subplots(2)
            fig.suptitle('Data before interpolation:')
            axs[0].plot(df['Timestamp'], df['Diameter'])
            axs[0].set(xlabel='Time (s)', ylabel='Diameter')
            axs[1].plot(df['Timestamp'], df['X'], label='x')
            axs[1].plot(df['Timestamp'], df['Y'], label='y')
            axs[1].set(xlabel='Time (s)', ylabel='Gaze coordinates')
            axs[1].legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust the subplot layout
            plt.show()

        # Create new dataframe to store the interpolated data
        df_interpolated = df.copy()

        # 1. Search for undetected blinks and set them to NaN
        # Find the rows where the diameter is below 0 and set the corresponding rows in the Blink column to 1
        df_interpolated.loc[df_interpolated['Diameter'] <= diameter_threshold , 'Blink'] = 1

        # 2. Merge two blinks into 1 if they are < coalesce togehter (coalesce blinks)
        # That is if the start of the next blink is within the coalesce duration of the end of the previous blink. Therefore, iterate through the dataframe.
        # If the rows containing a 1 are less than xy ms apart, merge the blinks by inserting a 1 in the rows between.
        blink_indices = df_interpolated.index[df_interpolated['Blink'] == 1].tolist()
        for i in range(1, len(blink_indices)):
            if df_interpolated['Timestamp'][blink_indices[i]] - df_interpolated['Timestamp'][blink_indices[i-1]] < config["coalesce"]:
                df_interpolated.loc[blink_indices[i-1]:blink_indices[i], 'Blink'] = 1
       
        # 3. Pad the blinks
        # Start of blinks are the rows containing 1 in the Blink column, following a row containing 0
        blink_starts = df_interpolated[(df_interpolated['Blink'] == 1) & (df_interpolated['Blink'].shift(1) == 0)].index
        # End of blinks are the rows containing 1 in the Blink column, following a row containing 0
        blink_ends = df_interpolated[(df_interpolated['Blink'] == 1) & (df_interpolated['Blink'].shift(-1) == 0)].index
        # Set the rows before and after blinks to 1
        for idx in blink_starts:
            df_interpolated.loc[max(0, idx - int(blink_padding)*sampling_rate):idx, 'Blink'] = 1
        for idx in blink_ends:
            df_interpolated.loc[idx:min(len(df_interpolated) - 1, idx + int(blink_padding)*sampling_rate), 'Blink'] = 1
       
        # Set all the values in the X, Y and Diameter columns to NaN where the Blink column contains 1
        df_interpolated.loc[df_interpolated['Blink'] == 1, ['X', 'Y', 'Diameter']] = np.nan

        # 4. Set saccades to NaN and pad them
        saccade_starts = df_interpolated[(df_interpolated['Saccade'] == 1) & (df_interpolated['Saccade'].shift(1) == 0)].index
        saccade_ends = df_interpolated[(df_interpolated['Saccade'] == 1) & (df_interpolated['Saccade'].shift(-1) == 0)].index
        for idx in saccade_starts:
            df_interpolated.loc[max(0, idx - int(saccade_padding)*sampling_rate):idx, 'Saccade'] = 1
        for idx in saccade_ends:
            df_interpolated.loc[idx:min(len(df_interpolated) - 1, idx + int(saccade_padding)*sampling_rate), 'Saccade'] = 1
        df_interpolated.loc[df_interpolated['Saccade'] == 1, ['X', 'Y', 'Diameter']] = np.nan

        # 5. Find values beyond the screen and set them to NaN. Subtract offset from screen resolution.
        screen_left = 0 - x_offset
        screen_right = screen_resolution[0] / 2 - x_offset
        screen_top = 0 - y_offset
        screen_bottom = screen_resolution[1] - y_offset
        
        # Identify out-of-bounds values and set them to NaN
        df_interpolated.loc[(df_interpolated['X'] < screen_left) | (df_interpolated['X'] > screen_right), 'X'] = np.nan
        df_interpolated.loc[(df_interpolated['Y'] < screen_top) | (df_interpolated['Y'] > screen_bottom), 'Y'] = np.nan
        
        # 6. Detect extreme values for diameter and set to NaN (median +/- 3 std)
        diameter_median = df_interpolated['Diameter'].median()
        diameter_std = df_interpolated['Diameter'].std()
        df_interpolated.loc[df_interpolated['Diameter'] < (diameter_median - 3 * diameter_std), 'Diameter'] = np.nan
        df_interpolated.loc[df_interpolated['Diameter'] > (diameter_median + 3 * diameter_std), 'Diameter'] = np.nan

        # 7. Interpolate NaN values
        df_interpolated['Diameter'] = df_interpolated['Diameter'].interpolate(method='linear', limit_direction='both')
        df_interpolated['X'] = df_interpolated['X'].interpolate(method='linear', limit_direction='both')
        df_interpolated['Y'] = df_interpolated['Y'].interpolate(method='linear', limit_direction='both')

        if plot:
            # Plot the data after interpolation. Diameter and gaze separately.
            fig, axs = plt.subplots(2)
            fig.suptitle('Data after interpolation:')
            axs[0].plot(df_interpolated['Timestamp'], df_interpolated['Diameter'])
            axs[0].set(xlabel='Time (s)', ylabel='Diameter')
            axs[1].plot(df_interpolated['Timestamp'], df_interpolated['X'], label='x')
            axs[1].plot(df_interpolated['Timestamp'], df_interpolated['Y'], label='y')
            axs[1].set(xlabel='Time (s)', ylabel='Gaze coordinates')
            axs[1].legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the subplot layout
            plt.show()

        return df_interpolated
    
    except Exception as e:
        raise Warning(f'Could not interpolate blinks properly! Error: {e}')
    


def filter_gaze_data(df, sampling_rate, lp, butter_order,plot=False):
    """ 
    Filter the gaze data using a low-pass butterworth filter.
    
    input:
        df: pd.DataFrame, dataframe containing the data
        sampling_rate: float, sampling rate of the data
        lp: float, low-pass frequency
        butter_order: int, order of the butterworth filter
        plot: bool, plot the data before and after filtering (default: False)

    output:
        df_filtered: pd.DataFrame, dataframe containing the data after filtering
    """ 
    try: 
        # Create new dataframe to store the filtered data
        df_filtered = df.copy()
     
        # Apply low-pass filter
        b, a = butter(butter_order, lp / (sampling_rate / 2), btype='low')
        df_filtered['X'] = filtfilt(b, a, df_filtered['X'])
        df_filtered['Y'] = filtfilt(b, a, df_filtered['Y'])

        # Plot the data before and after filtering in subplots
        if plot:
            fig, axs = plt.subplots(2)
            fig.suptitle('Data before and after filtering:')
            axs[0].plot(df['Timestamp'], df['X'], label='x')
            axs[0].plot(df['Timestamp'], df['Y'], label='y')
            axs[0].set(xlabel='Time (s)', ylabel='Gaze coordinates')
            axs[0].legend(loc='upper right')
            axs[1].plot(df_filtered['Timestamp'], df_filtered['X'], label='x_filtered')
            axs[1].plot(df_filtered['Timestamp'], df_filtered['Y'], label='y_filtered')
            axs[1].set(xlabel='Time (s)', ylabel='Gaze coordinates')
            axs[1].legend(loc='upper right')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the subplot layout
            plt.show()

        return df_filtered
    
    except Exception as e:
        raise Warning(f'Could not filter gaze data properly! Error: {e}')
    


def calculate_centroid(data, threshold):
    """
    Calculate the ceneteroid of the gaze data, without outliers
    
    Input: 
    - data: DataFrame with gaze data
    - threshold: Threshold for the distance of each point to the centroid
    
    Output:
    - centroid: List with x and y coordinates of the centroid
    """

    # Calculate the centroid of the gaze data
    x = data['X']
    y = data['Y']
    centroid = [x.mean(), y.mean()]

    # Calculate the distance of each point to the centroid
    distances = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)
    
    # Remove outliers
    data = data[distances < threshold]
    
    # Calculate the centroid again
    x = data['X']
    y = data['Y']
    centroid = [x.mean(), y.mean()]
    
    return centroid


def align_data(df, center, threshold, plot=False):
    """
    Align the data to the center of the screen.

    Input:
        df: pd.DataFrame, dataframe containing the data
        center: list, center of the screen in pixels [x, y]
        threshold: float, threshold for the distance of each point to the centroid
        plot: bool, plot the data before and after alignment (default: False)

    Output:
        df_aligned: pd.DataFrame, dataframe containing the aligned data
    """

    try: 
        # Make a copy of the dataframe
        df_aligned = df.copy()
        
        # Only use fixation data for alignment
        df_fixations = df_aligned[df_aligned['Fixation'] == 1].copy()

        # Calculate the centroid of the gaze data
        centroid = calculate_centroid(df_fixations, threshold)

        # Plot the gaze data before alignment
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(df_fixations['X'], df_fixations['Y'], s=1, label='Gaze data')
            plt.plot(center_x, center_y, 'k+', markersize=6, label='Center')
            plt.plot(centroid[0], centroid[1], 'ro', label='Centroid')
            # Plot threshold circle
            circle = plt.Circle(centroid, threshold, color='r', fill=False, linestyle='--', linewidth=1, label='Threshold')
            plt.gca().add_artist(circle)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().invert_yaxis()  # Invert the y-axis to have the origin in the top left corner
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()
        
        # Calculate the difference between the centroid and the center
        diff = [center[0] - centroid[0], center[1] - centroid[1]]

        # Align the gaze data to the center
        df_aligned['X'] = df_aligned['X'] + diff[0]
        df_aligned['Y'] = df_aligned['Y'] + diff[1]

        # Calculate the new centroid
        new_centroid = calculate_centroid(df_aligned, threshold)

        # Get fixation data
        df_fixations_aligned = df_aligned[df_aligned['Fixation'] == 1]

        # Plot the gaze data after alignment
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(df_fixations_aligned['X'], df_fixations_aligned['Y'], s=1, label='Gaze data')
            plt.plot(center_x, center_y, 'k+', markersize=6, label='Center')
            plt.plot(new_centroid[0], new_centroid[1], 'ro', label='Centroid')
            # Plot threshold circle
            circle = plt.Circle(new_centroid, threshold, color='r', fill=False, linestyle='--', linewidth=1, label='Threshold')
            plt.gca().add_artist(circle)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.gca().invert_yaxis()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()

        return df_aligned
    except Exception as e:
        raise Warning(f'Could not align the data properly! Returned unaligned data. Error: {e}')
    
def exclude_trials(df, missing_data_threshold):
    """
    Exclude trials in which more than 30% of the data is missing
    
    input:
        df: pd.DataFrame, dataframe containing the data
        missing_data_threshold: float, threshold for missing data

    output:
        df: pd.DataFrame, dataframe containing the data after exclusion
    """
    try:
        # Calculate the percentage of missing data for each trial (= number of columns with blink values set to 1)
        trial_lengths = df['Trial'].value_counts()
        missing_data = df.groupby('Trial')['Blink'].sum() / trial_lengths
        #print(f'The percentage of missing data for each trial is: {missing_data}')

        # Exclude trials in which more than 30% of the data is missing
        trials_to_exclude = missing_data[missing_data > missing_data_threshold].index
        df = df[~df['Trial'].isin(trials_to_exclude)]

        # Print names of trials that were excluded
        if len(trials_to_exclude) > 0:
            print(f'The following trials were excluded because more than {missing_data_threshold*100}% of the data was missing: {trials_to_exclude.tolist()}')
        else:
            print(f'No trials were excluded because more than {missing_data_threshold*100}% of the data was missing.') 
        
        return df
    except Exception as e:
        raise Warning(f'Could not exclude data properly! Error: {e}')


def process_file(file, directory=""):

    # Extract the run number and report
    pattern = re.compile(r's(\d{2})r(\d{2})(nr|r)')
    subject_nr, run_nr, report = pattern.match(file).groups()
    print(f'Subject number: {subject_nr}, Run number: {run_nr}, Report: {report}')

    # 1. Read in the data and return the dataframe
    #print(f'Step 1: Reading in the data.')
    df, gaze_coords, sampling_rate, n_saccades, n_fixations, n_blinks = read_eyelink_data(directory + "/" + file)

    # Check that no timestamps are missing
    missing = False
    for i in range(len(df) - 1):
        if round(df['Timestamp'][i+1]*sampling_rate - df['Timestamp'][i]*sampling_rate) != 1:
            missing = True
            print('Missing timestamp at index: ' + str(df['Timestamp'][i]))
    if not missing:
        print('No missing timestamps.')

    # Read in .mat file that contains the word "stimuli"
    parameters_files = [f for f in os.listdir(directory)  if f.endswith('.mat') and 'stimuli' in f]

    # Check if there is a stimuli file
    if len(parameters_files) == 0:
        raise ValueError('No stimuli file found.')
    elif len(parameters_files) > 1:
        # Search for file that contains the same run number after "run"
        closest_run_nr = -1
        for parameters_file in parameters_files:
            # Extract run number from the file name
            parameters_file_run_nr = re.search(r'run(\d+)', parameters_file).group(1)
            if int(parameters_file_run_nr) >= int(closest_run_nr) and int(parameters_file_run_nr) <= int(run_nr):
                closest_run_nr = parameters_file_run_nr
                closest_file = parameters_file
        print(f'Closest run number: {closest_run_nr}')
        parameters_file = loadmat(directory + '/' + closest_file)
    else:
        parameters_file = loadmat(directory + '/' + parameters_files[0])
    
    # Extract offset
    stimuli = parameters_file['stimuli']
    x_offset = stimuli["xOffset"][0][0][0][0]
    y_offset = stimuli["yOffset"][0][0][0][0]
    
    # Get the number of trials
    n_trials = max(df['Trial'].dropna().astype(int))

    for align in [True, False]:
        if align:
            print('\nAlignment: On')
        else:
            print('\nAlignment: Off')

        # Iterate over all trials
        for trial in range(1, n_trials + 1):

            # Extract the data for the respective trial
            trial_data = df[df['Trial'] == str(trial)].copy()
            # Trial starts with trial countdown. Therefore, the onset of the first fixation is the more precise start of the trial.
            trial_data = trial_data[trial_data['Fixpoint1'].notna()]
            # Set the time to 0 for the beginning of the trial
            time_start = trial_data['Timestamp'].min()
            trial_data['Timestamp'] = (trial_data['Timestamp'] - time_start)

            # 2. Interpolate blinks
            df_interpolated = interpolate_blinks(trial_data, config["blink_padding"], config["saccade_padding"], config["diameter_threshold"], config["plot"], sampling_rate, config["screen_resolution"], x_offset, y_offset, trial)

            # 3. Filter x and y coordinates
            df_filtered = filter_gaze_data(df_interpolated, sampling_rate, config["lp"], config["butter_order"], config["plot"])

            # 4. Align data
            if align:
                df_aligned = align_data(df_filtered, center, threshold, config["plot"])
            else: 
                df_aligned = df_filtered

            # Set trials together to whole dataframe
            if trial == 1:
                df_preprocessed = df_aligned
            else:
                df_preprocessed = pd.concat([df_preprocessed, df_aligned])

        # 5. Exclude data
        df_preprocessed = exclude_trials(df_preprocessed, config["missing_data_threshold"])
        if len(df_preprocessed) == 0:
            print('Whole run is excluded due to missing data!')
            continue
        else:
            # 6. Save the data
            if align:
                df_preprocessed.to_csv(os.path.join(directory, f's{subject_nr}r{run_nr}{report}_preprocessed.csv'), index=False)
            else:
                df_preprocessed.to_csv(os.path.join(directory, f's{subject_nr}r{run_nr}{report}_preprocessed_not_aligned.csv'), index=False)
        
    print(f'\nPreprocessing data for file {file} completed.')


def process_folder(directory, excluded_runs):

    # Get all ".asc" files
    files = os.listdir(directory)
    files = [file for file in files if file.endswith(".asc")]

    # Sort files by run number
    files.sort()

    # Exclude runs (file format: "s00r00r.asc") Check if number after r is in excluded_runs
    run_number_pattern = re.compile(r's\d+r(\d+)(r|nr)\.asc')
    print(f'Found eyelink files: {files}')
    files = [file for file in files if (match := run_number_pattern.search(file)) and int(match.group(1)) not in excluded_runs]
    print(f'Excluded runs: {excluded_runs}')

    ## Start preprocessing pipeline for each file
    # Iterate over all files
    for file in files:
        print(f'\nPreprocessing data for file {file}.')
        process_file(file, str(directory))




if __name__ == "__main__":

    #load configurations
    config = load_config()
    center = [config["screen_resolution"][0] / 4, config["screen_resolution"][1] / 2] # Center of the stimulus on the left half of the screen
    center_x, center_y = center

    # Define the threshold for outlier removal when calculating the centroid of the fixation data 
    threshold = np.sqrt(2 * config["distance"]**2)*2 # Twice the distance of the diagonal fixation spots to the center in pixels


    print(f"Plotting the data: {config['plot']}")
    print(f"Coalescing blinks with a distance of less than {config['coalesce']}s.")
    print(f"Using a threshold of {config['diameter_threshold']} for the diameter of the pupil.")
    print(f"Using a padding value of: {config['blink_padding']}s for blinks and {config['saccade_padding']}s for saccades.")
    print(f"Using a low-pass frequency of {config['lp']} Hz and a butterworth filter of order {config['butter_order']}.")
    print(f"Screen width: {config['screen_width']}mm, Screen height: {config['screen_height']}mm, Screen distance: {config['screen_distance']}mm, Screen resolution: {config['screen_resolution']} pixels")
    print(f"Center of the stimulus: {center}")
    print(f"Distance between the fixation spots to center: {config['distance']} pixels")
    print(f"Using a threshold of {threshold} for fixation data used for alignment.")
    print(f"Using a threshold of {config['missing_data_threshold']*100}% for missing data.")


    parser = argparse.ArgumentParser(description="Analysis of ASC files")
    parser.add_argument("path", type=Path, help="Path to a file or a directory")

    parser.add_argument("--exclude", type=int, nargs='+', default=[], help="List of runs to exclude (e.g., --exclude 1 3 5)")
    args = parser.parse_args()

    if args.path.is_file():
        filename = args.path.name
        directory = args.path.parent
        process_file(filename, str(directory))

    elif args.path.is_dir():
        process_folder(args.path, args.exclude)
    else:
        print(f"invalid path: {args.path}")


#'todo, subject nr in process folder behalten?' 
#subject nummer als argument oder aus ordner name extrahieren