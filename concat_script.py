import os
import argparse
import pandas as pd

def concatenate_csvs(root_dir, output_dir):
    # Define the GPU types
    gpu_types = ['a100', 'v100']

    # Define the CSV file names
    csv_files_names = ['jaxdecompfft.csv', 'jaxfft.csv', 'mpi4jax.csv']

    # Iterate over each GPU type
    for gpu in gpu_types:
        gpu_dir = os.path.join(root_dir, gpu)
        
        # Check if the GPU directory exists
        if not os.path.exists(gpu_dir):
            continue
        
        for csv_file_name in csv_files_names:
            # List CSV in directory and subdirectories
            csv_files = []
            for root, dirs, files in os.walk(gpu_dir):
                for file in files:
                    if file.endswith(csv_file_name):
                        csv_files.append(os.path.join(root, file))

            # Concatenate CSV files
            combined_df = pd.DataFrame()
            for csv_file in csv_files:
                print(f'Concatenating {csv_file}...')
                df = pd.read_csv(csv_file,
                                 header=None,
                                 names=[
                                     "rank", "FFT_type", "precision", "x", "y", "z",
                                     "px", "py", "backend", "nodes", "time"
                                 ],
                                 index_col=False)
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            # Remove duplicates based on specified columns
            combined_df.drop_duplicates(
                subset=[
                    "rank", "FFT_type", "precision", "x", "y", "z", "px", "py",
                    "backend", "nodes"
                ],
                keep='last',
                inplace=True
            )

            if not os.path.exists(os.path.join(output_dir, gpu)):
                print(f"Creating directory {os.path.join(output_dir, gpu)}")
                os.makedirs(os.path.join(output_dir, gpu))

            output_file = os.path.join(output_dir, gpu, csv_file_name)
            combined_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate CSV files and remove duplicates by GPU type.')
    parser.add_argument('-i', '--input', dest='root_dir', type=str, help='Root directory containing CSV files.', required=True)
    parser.add_argument('-o', '--output', dest='output_dir', type=str, help='Output directory to save concatenated CSV files.', required=True)
    
    args = parser.parse_args()
    
    concatenate_csvs(args.root_dir, args.output_dir)
