# utils.py
import numpy as np

def extract_traces_from_patches(patches_data, num_traces_per_patch=5, seed=None):
    """Extracts traces from 3D patches.

    Args:
        patches_data (np.ndarray): The 3D patches (N, C, D, H, W).
        num_traces_per_patch (int): Number of random traces to extract and average per patch.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Extracted traces (N, D).
    """
    if seed is not None:
        np.random.seed(seed)
        
    N, C, D, H, W = patches_data.shape
    all_traces = np.zeros((N, D), dtype=np.float32)
    
    for i in range(N):
        if num_traces_per_patch > 0 and H > 0 and W > 0:
            patch_traces_list = np.zeros((num_traces_per_patch, D), dtype=np.float32)
            for j in range(num_traces_per_patch):
                # Ensure h, w are valid indices
                rand_h = np.random.randint(0, H) if H > 0 else 0
                rand_w = np.random.randint(0, W) if W > 0 else 0
                # Assuming the actual seismic data is in the first channel (C=0)
                patch_traces_list[j] = patches_data[i, 0, :, rand_h, rand_w]
            all_traces[i] = np.mean(patch_traces_list, axis=0)
        elif D > 0 and H > 0 and W > 0: # Fallback if num_traces_per_patch is 0, take center trace or first trace
            center_h, center_w = H // 2, W // 2
            all_traces[i] = patches_data[i, 0, :, center_h, center_w]
        else: # If patch dimensions are problematic, return zeros for this trace
            all_traces[i] = np.zeros(D, dtype=np.float32)
            
    return all_traces

def extract_random_trace_from_patch(patch_data_single):
    """Extracts a single random trace from a single 3D patch.
    Note: The original notebook called a similar function 'extract_center_trace',
    but it selected a random H, W. This function does the same for consistency.

    Args:
        patch_data_single (np.ndarray): A single 3D patch (C, D, H, W).

    Returns:
        np.ndarray: Extracted trace (D,).
    """
    C, D, H, W = patch_data_single.shape
    if H == 0 or W == 0: # Handle cases with zero spatial dimensions
        return np.zeros(D, dtype=np.float32)
        
    random_h = np.random.randint(0, H)
    random_w = np.random.randint(0, W)
    # Assuming the actual seismic data is in the first channel (C=0)
    trace = patch_data_single[0, :, random_h, random_w]
    return trace.astype(np.float32)

if __name__ == '__main__':
    # Example Usage
    print("Testing utility functions...")
    dummy_patches = np.random.rand(2, 1, 32, 32, 32).astype(np.float32) # N=2, C=1, D=32, H=32, W=32
    
    # Test extract_traces_from_patches
    extracted_avg_traces = extract_traces_from_patches(dummy_patches, num_traces_per_patch=5, seed=42)
    print(f"Shape of extracted average traces: {extracted_avg_traces.shape}") # Expected (2, 32)

    # Test extract_random_trace_from_patch
    single_patch_for_random_trace = dummy_patches[0]
    random_trace = extract_random_trace_from_patch(single_patch_for_random_trace)
    print(f"Shape of single extracted random trace: {random_trace.shape}") # Expected (32,)

    print("Utility functions test complete.")

