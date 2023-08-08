import pandas as pd
import pickle
import os

def save_selected_columns_to_pickle(input_csv, output_pickle, selected_columns):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Select only the columns you want to keep
    df_selected = df[selected_columns]

    # Save the selected data to a pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(df_selected, f)

input_csv = 'pitches.csv'
output_pickle = 'pitch_identification.pkl'
selected_columns = ['px', 'pz', 'start_speed', 'end_speed', 'spin_rate', 'spin_dir', 'ax', 'ay', 'az', 'vx0', 'vy0', 'vz0', 'break_angle', 'break_length', 'break_y', 'pitch_type']  # Replace with your desired column names

# Get absolute paths to handle different working directories
input_csv = os.path.abspath(input_csv)
output_pickle = os.path.abspath(output_pickle)

# Check if the input CSV file exists
if not os.path.exists(input_csv):
    print(f"Error: Input CSV file '{input_csv}' not found.")
else:
    save_selected_columns_to_pickle(input_csv, output_pickle, selected_columns)
    print(f"Selected columns saved to '{output_pickle}'.")

def view_data_from_pickle(pickle_file, num_rows=5):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Assuming the last column is the target label
    labels = data.iloc[:, -1]
    data = data.iloc[:, :-1]

    print("Data:")
    print(data.head(num_rows))

    print("\nLabels:")
    print(labels.head(num_rows))

# # Example usage:
# # Replace 'data.pickle' with the path to your pickle file
# input_pickle_file = 'pitch_identification.pickle'
# view_data_from_pickle(input_pickle_file)

def extract_max_min_values(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)


    data = data.iloc[1:, :-1]
    # Initialize dictionaries to store max and min values for each feature
    max_values = {}
    min_values = {}

    # Iterate over the columns (features) in the data
    for column in data.columns:
        # Extract the maximum and minimum values for the current feature
        max_values[column] = data[column].max()
        min_values[column] = data[column].min()

    return max_values, min_values

# Example usage:
# Replace 'data.pickle' with the path to your pickle file
# input_pickle_file = 'pitch_identification_nonan.pkl'
# max_values, min_values = extract_max_min_values(input_pickle_file)

# # Display max and min values for each feature
# for column in max_values:
#     print(f"Feature: {column}, Max Value: {max_values[column]}, Min Value: {min_values[column]}")


def remove_rows_with_nan_and_save(input_file, output_file):
    # Load the pickle file into a pandas DataFrame
    data = pd.read_pickle(input_file)

    #TODO: remove rows where last column value (pitch_type) is "PO" "FO" "UN" "IN"
    data = data[~data['pitch_type'].isin(["PO", "FO", "UN", "IN"])]

    # Filter out rows with NaN in the "pitch_type" column
    data = data.dropna()

    # Save the filtered data as a new pickle file
    data.to_pickle(output_file)

# Example usage:
input_filename = "pitch_identification.pkl"
output_filename = "pitch_identification_nonan.pkl"
remove_rows_with_nan_and_save(input_filename, output_filename)