import pandas as pd
import numpy as np
import cupy as cp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # For tracking progress

# Parameters
n_iterations = 1000  # Number of Monte Carlo iterations
k_values = range(1, 41)  # Reduced range of k values to evaluate to save memory
batch_size = 1000  # Batch size for distance calculation

def load_and_clean_data(filepath):
    """
    Load dataset and clean unnecessary columns and duplicates.
    """
    # Load the dataset
    df = pd.read_csv(filepath)

    # Remove rows with duplicate compositions, keeping only the first occurrence
    df = df.drop_duplicates(subset='composition', keep='first').reset_index(drop=True)

    # Drop unnecessary columns
    columns_to_drop = ['composition', 'sample_id', 'prop_x', 'prop_y', 'unit_x', 'unit_y', 
                       'Temperature', 'Thermal Conductivity', 'comp_obj', 'atomic_fractions', 
                       'group', 'class']
    df = df.drop(columns=columns_to_drop, errors='ignore').reset_index(drop=True)

    return df

def monte_carlo_iteration(iteration, df, k_values):
    """
    Perform one Monte Carlo iteration, including splitting data, scaling,
    calculating distances, and determining applicability domain.
    """
    # Step 1: Split the data into 80% training and 20% testing data
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=np.random.randint(0, 10000))

    # Step 1.1: Apply StandardScaler to the training and test data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    # Convert training and test data to CuPy arrays for GPU acceleration
    train_data_gpu = cp.array(train_data_scaled, dtype=cp.float32)
    test_data_gpu = cp.array(test_data_scaled, dtype=cp.float32)

    # Initialize a dictionary to store results for the current iteration
    iteration_results = {k: {'within_domain': 0, 'out_of_domain': 0} for k in k_values}
    
    # Step 2: Calculate the entire distance matrix between all training samples in batches to save memory
    n_train = train_data_gpu.shape[0]
    full_distances = cp.full((n_train, n_train), cp.nan, dtype=cp.float32)

    for start_idx in range(0, n_train, batch_size):
        end_idx = min(start_idx + batch_size, n_train)
        full_distances[start_idx:end_idx] = cp.linalg.norm(train_data_gpu[start_idx:end_idx, cp.newaxis, :] - train_data_gpu[cp.newaxis, :, :], axis=2)

    cp.get_default_memory_pool().free_all_blocks()  # Free GPU memory after calculation

    # Step 3: Exclude self-distances (set them to NaN)
    cp.fill_diagonal(full_distances, cp.nan)

    # Loop through each k value to calculate applicability domain
    for k in k_values:
        avg_knn_distances = cp.nanmean(cp.sort(full_distances, axis=1)[:, :k], axis=1)

        # Step 5: Calculate reference value (Ref Val) using Q1 and Q3 of avg_knn_distances
        Q1 = cp.percentile(avg_knn_distances, 25)
        Q3 = cp.percentile(avg_knn_distances, 75)
        IQR = Q3 - Q1
        reference_value = Q3 + 1.5 * IQR

        # Step 6: Filter the initial distance matrix based on the reference value to define the neighborhood
        filtered_distances = cp.where(full_distances <= reference_value, full_distances, cp.nan)
        Ki_values = cp.sum(~cp.isnan(filtered_distances), axis=1)

        # Step 7: Calculate the threshold ti for each training sample
        thresholds = cp.full(filtered_distances.shape[0], cp.nan, dtype=cp.float32)
        for i in range(filtered_distances.shape[0]):
            if Ki_values[i] > 0:
                thresholds[i] = cp.nanmean(filtered_distances[i, :])

        min_threshold = cp.nanmin(thresholds[~cp.isnan(thresholds)])
        thresholds = cp.where(cp.isnan(thresholds), min_threshold, thresholds)

        # Step 8: Evaluate the applicability domain on the test data (20%)
        test_distances = cp.linalg.norm(test_data_gpu[:, cp.newaxis, :] - train_data_gpu[cp.newaxis, :, :], axis=2)
        within_domain_mask = cp.any(test_distances <= thresholds[cp.newaxis, :], axis=1)
        within_domain_count = cp.sum(within_domain_mask)
        out_of_domain_count = test_data_gpu.shape[0] - within_domain_count

        # Update iteration results
        iteration_results[k]['within_domain'] = int(within_domain_count)
        iteration_results[k]['out_of_domain'] = int(out_of_domain_count)

    # Free GPU memory after each iteration
    del full_distances, test_data_gpu, train_data_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return iteration_results

def main():
    # Load and clean the data
    df = load_and_clean_data("../training_data.csv")

    # Run Monte Carlo iterations
    all_iterations_results = []
    for iteration in tqdm(range(n_iterations), desc="Monte Carlo Iterations"):
        result = monte_carlo_iteration(iteration, df, k_values)
        all_iterations_results.append(result)

    # Store iteration details for each iteration separately
    iteration_details = {k: [] for k in k_values}
    for iteration_idx, result in enumerate(all_iterations_results):
        for k in k_values:
            iteration_details[k].append({'iteration': iteration_idx, 'within_domain': result[k]['within_domain'], 'out_of_domain': result[k]['out_of_domain']})

    # Save iteration details to a CSV file
    iteration_details_list = []
    for k, details in iteration_details.items():
        for detail in details:
            iteration_details_list.append({'k': k, 'iteration': detail['iteration'], 'within_domain': detail['within_domain'], 'out_of_domain': detail['out_of_domain']})
    iteration_details_df = pd.DataFrame(iteration_details_list)
    iteration_details_df.to_csv("knn_iteration_details.csv", index=False)

if __name__ == "__main__":
    main()
