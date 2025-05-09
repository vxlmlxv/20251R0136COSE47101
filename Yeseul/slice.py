import pandas as pd
import os

def slice_csv(file_path, output_dir, max_size_mb=30):
    """
    Slices a CSV file into smaller chunks based on a maximum size limit.

    Args:
        file_path (str): Path to the input CSV file.
        output_dir (str): Directory to store the sliced CSV files.
        max_size_mb (int): Maximum size of each slice in megabytes.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    chunk_size = 10000  # Adjust as needed for optimal performance
    file_size_bytes = os.path.getsize(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size_bytes <= max_size_bytes:
        print(f"File size ({file_size_bytes} bytes) is already within the limit ({max_size_bytes} bytes). No slicing needed.")
        return

    total_rows = sum(1 for _ in open(file_path, 'r')) -1 # Subtract header row
    num_chunks = (file_size_bytes // max_size_bytes) + 1
    rows_per_chunk = total_rows // num_chunks

    print(f"Total rows: {total_rows}, Number of chunks: {num_chunks}, Rows per chunk: {rows_per_chunk}")

    chunk_num = 0
    for chunk in pd.read_csv(file_path, chunksize=rows_per_chunk):
        output_file = os.path.join(output_dir, f"slice_{chunk_num}.csv")
        chunk.to_csv(output_file, index=False)
        print(f"Created: {output_file} (shape: {chunk.shape})")
        chunk_num += 1


if __name__ == "__main__":
    input_file = "reports.csv"  # Replace with your actual file path
    output_directory = "sliced_data"
    slice_csv(input_file, output_directory)
