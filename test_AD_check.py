import pandas as pd
import numpy as np
import cupy as cp
from joblib import load
from tqdm import tqdm
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition.composite import ElementProperty

def load_and_preprocess_data(file_path):
    # Load the dataset from CSV file
    df = pd.read_csv(file_path)
    # Drop the 'composition' column as it will not be used in distance calculations
    df_no_composition = df.drop(['composition'], axis=1)
    return df, df_no_composition

def convert_to_gpu_array(df_no_composition):
    # Convert the dataframe to a CuPy GPU array for faster calculations
    return cp.array(df_no_composition.values, dtype=cp.float32)

def compute_blockwise_distance_matrix(df_gpu, block_size):
    # Get the number of samples
    n_samples = df_gpu.shape[0]
    # Initialize the full distance matrix with NaN values
    full_distances = cp.full((n_samples, n_samples), cp.nan, dtype=cp.float32)
    # Calculate distances in blocks to manage memory usage
    for i in range(0, n_samples, block_size):
        for j in range(0, n_samples, block_size):
            # Extract blocks of data for pairwise distance calculation
            block_1 = df_gpu[i:i + block_size]
            block_2 = df_gpu[j:j + block_size]
            # Compute the pairwise Euclidean distance between two blocks
            distances_block = cp.linalg.norm(block_1[:, cp.newaxis, :] - block_2[cp.newaxis, :, :], axis=2)
            # Fill the appropriate part of the full distance matrix
            full_distances[i:i + block_size, j:j + block_size] = distances_block
    return full_distances

def calculate_knn_thresholds(full_distances, k):
    # Set diagonal distances (self-distances) to NaN
    cp.fill_diagonal(full_distances, cp.nan)
    # Sort distances for each sample to find the k-nearest neighbors
    sorted_distances = cp.sort(full_distances, axis=1)
    k_nearest_distances = sorted_distances[:, :k]
    # Calculate the average distance to the k-nearest neighbors for each sample
    avg_knn_distances = cp.nanmean(k_nearest_distances, axis=1)

    # Calculate the first (Q1) and third (Q3) quartiles and the interquartile range (IQR)
    Q1, Q3 = cp.percentile(avg_knn_distances, [25, 75])
    IQR = Q3 - Q1
    # Define the reference value for identifying outliers
    reference_value = Q3 + 1.5 * IQR

    # Filter distances to only include those below the reference value (for density estimation)
    filtered_distances = cp.where(full_distances <= reference_value, full_distances, cp.nan)
    # Count the number of neighbors within the reference value for each sample
    Ki_values = cp.sum(~cp.isnan(filtered_distances), axis=1)

    # Initialize thresholds with NaN values
    thresholds = cp.full(filtered_distances.shape[0], cp.nan, dtype=cp.float32)
    # Calculate the threshold for each sample where there are valid neighbors
    for i in range(filtered_distances.shape[0]):
        if Ki_values[i] > 0:
            thresholds[i] = cp.nanmean(filtered_distances[i, :])

    # Replace NaN thresholds with the minimum valid threshold
    min_threshold = cp.nanmin(thresholds[~cp.isnan(thresholds)])
    thresholds = cp.where(cp.isnan(thresholds), min_threshold, thresholds)
    return thresholds

def save_thresholds(thresholds, file_path):
    # Move the thresholds from GPU to CPU for saving
    thresholds_cpu = thresholds.get()
    # Save thresholds to a CSV file
    thresholds_df = pd.DataFrame({'sample_index': np.arange(len(thresholds_cpu)), 'threshold': thresholds_cpu})
    thresholds_df.to_csv(file_path, index=False)
    print(f"Thresholds calculated and saved to '{file_path}'")

def prepare_test_sample(material):
    # Create a new DataFrame for the test material
    df_new = pd.DataFrame({'material': [material]})
    # Convert material strings to composition objects
    df_new = StrToComposition().featurize_dataframe(df_new, "material", ignore_errors=True)
    # Apply Magpie feature set after converting to composition
    magpie_featurizer = ElementProperty.from_preset("magpie")
    df_new = magpie_featurizer.featurize_dataframe(df_new, "composition", ignore_errors=True)
    # Drop unnecessary columns ('material' and 'composition')
    df_new = df_new.drop(['material', 'composition'], axis=1)
    return df_new

def scale_test_sample(df_new, scaler_path, df_no_composition_columns):
    # Load the scaler to apply the same scaling as the training data
    scaler = load(scaler_path)
    # Scale the new data
    u2ti_scaled = scaler.transform(df_new)
    # Match the columns to the original training data structure
    df_u2ti = pd.DataFrame(u2ti_scaled, columns=df_new.columns)[df_no_composition_columns]
    return df_u2ti

def evaluate_applicability_domain(df_u2ti, df_gpu, thresholds):
    # Convert the scaled test sample to a GPU array
    u2ti_scaled_gpu = cp.array(df_u2ti, dtype=cp.float32)
    # Calculate distances between the test sample and all training samples
    test_distances = cp.linalg.norm(u2ti_scaled_gpu - df_gpu, axis=1)
    # Determine how many training samples fall within the applicability domain of the test sample
    within_domain_mask = test_distances <= thresholds
    within_domain_count = cp.sum(within_domain_mask).get()
    print(f"The test sample falls within the applicability domain of {within_domain_count} training samples")
    return test_distances

def find_closest_sample(test_distances, df, df_no_composition, df_u2ti):
    # Find the index of the closest training sample to the test sample
    closest_sample_idx = cp.argmin(test_distances).get()
    # Retrieve the composition of the closest training sample
    closest_composition = df.loc[closest_sample_idx, "composition"]
    print(f"Closest composition for U2Ti in training data: {closest_composition}")
    # Get feature values of the closest training sample
    closest_sample_features = df_no_composition.iloc[closest_sample_idx].values
    # Convert the test sample to CPU for comparison
    u2ti_scaled_cpu = df_u2ti.values[0]
    # Calculate the absolute differences between the test sample and the closest training sample
    feature_differences = np.abs(u2ti_scaled_cpu - closest_sample_features)
    return feature_differences, df_no_composition.columns

def display_closest_features(feature_differences, feature_names, top_n=10):
    # Create a DataFrame to map feature differences to their respective names
    difference_df = pd.DataFrame({'Feature': feature_names, 'Difference': feature_differences})
    # Sort the features by their differences and reset index
    closest_features = difference_df.sort_values(by="Difference").reset_index(drop=True)
    # Display the top N features with the smallest differences for analysis
    print(f"Top {top_n} features with the smallest differences:")
    print(closest_features.head(top_n))

if __name__ == "__main__":
    # Load and preprocess data
    df, df_no_composition = load_and_preprocess_data("training_data_post_pearson.csv")
    df_gpu = convert_to_gpu_array(df_no_composition)

    # Compute distances and calculate thresholds
    full_distances = compute_blockwise_distance_matrix(df_gpu, block_size=1000)
    thresholds = calculate_knn_thresholds(full_distances, k=25)
    save_thresholds(thresholds, "knn_thresholds.csv")

    # Prepare and scale the test sample
    df_new = prepare_test_sample('U2Ti')
    df_u2ti = scale_test_sample(df_new, "scaler.pkl", df_no_composition.columns)

    # Evaluate applicability domain
    test_distances = evaluate_applicability_domain(df_u2ti, df_gpu, thresholds)

    # Find closest training sample and compare features
    feature_differences, feature_names = find_closest_sample(test_distances, df, df_no_composition, df_u2ti)
    display_closest_features(feature_differences, feature_names)
