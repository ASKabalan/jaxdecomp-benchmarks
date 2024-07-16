import os
import argparse

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
            all_lines = []
            for csv_file in csv_files:
                print(f'Concatenating {csv_file}...')
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    lines = [line for line in lines if line.strip()]
                    if len(lines) == 0:
                        continue
                    all_lines.extend(lines)

            if not os.path.exists(os.path.join(output_dir,gpu)):
                print(f"Creating directory {os.path.join(output_dir,gpu)}")
                os.makedirs(os.path.join(output_dir,gpu))
            output_file = os.path.join(output_dir,gpu, os.path.basename(csv_file))

            with open(output_file, 'a+') as f:
                f.write(''.join(all_lines))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate CSV files by GPU type.')
    parser.add_argument('-i', '--input', dest='root_dir', type=str, help='Root directory containing CSV files.', required=True)
    parser.add_argument('-o', '--output', dest='output_dir', type=str, help='Output directory to save concatenated CSV files.', required=True)
    
    args = parser.parse_args()
    
    concatenate_csvs(args.root_dir, args.output_dir)
