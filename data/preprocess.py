'''
Helper file to parse the large CSVs into pickle files
'''
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

def save_selected_columns_to_pickle(input_csv, output_pickle, selected_columns):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_csv)

    # Select only the columns you want to keep
    df_selected = df[selected_columns]

    # Save the selected data to a pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(df_selected, f)

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



def remove_rows_and_split(input_file, train_output_file, test_output_file):
    # Load the pickle file into a pandas DataFrame
    data = pd.read_pickle(input_file)

    # remove rows with pitchout (PO, FO) and unidentified (UN, IN)
    data = data[~data['pitch_type'].isin(["PO", "FO", "UN", "IN"])]

    # Filter out rows with NaN in the "pitch_type" column
    data = data.dropna()

    # split into train and test and save as 2 files
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    with open(train_output_file, 'wb') as f:
        pickle.dump(train_df, f)
    print(f"Processed training data saved to {train_output_file}")

    with open(test_output_file, 'wb') as f:
        pickle.dump(test_df, f)
    print(f"Processed test data saved to {test_output_file}")


def main():
    # CREATE PICKLE FROM CSV BY COLUMNS TO INCLUDE
    # selected_columns = ['px', 'pz', 'start_speed', 'end_speed', 'spin_rate', 'spin_dir', 'ax', 'ay', 'az', 'vx0', 'vy0', 'vz0', 'break_angle', 'break_length', 'break_y', 'pitch_type']  # Replace with your desired column names

    # # use absolute path
    # input_csv = os.path.abspath('pitches.csv')
    # output_pickle = os.path.abspath('pitch_identification.pkl')

    # # Check if the input CSV file exists
    # if not os.path.exists(input_csv):
    #     print(f"Error: Input CSV file '{input_csv}' not found.")
    # else:
    #     save_selected_columns_to_pickle(input_csv, output_pickle, selected_columns)
    #     print(f"Selected columns saved to '{output_pickle}'.")

    # VIEW DATA FROM PICKLE
    # input_pickle_file = 'pitch_identification.pkl'
    # view_data_from_pickle(input_pickle_file)

    # GET MIN AND MAX VALUES FOR NORMALIZATION
    # input_pickle_file = 'pitch_identification_nonan.pkl'
    # max_values, min_values = extract_max_min_values(input_pickle_file)

    # # Display max and min values for each feature
    # for column in max_values:
    #     print(f"Feature: {column}, Max Value: {max_values[column]}, Min Value: {min_values[column]}")

    # FILTER PICKLE AND SPLIT INTO TRAIN AND TEST
    remove_rows_and_split("pitch_identification.pkl", "pitch_identification_train.pkl", "pitch_identification_test.pkl")

if __name__ == "__main__":
    main()