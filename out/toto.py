import pandas as pd


def load_and_prepare_data(file_path):
  # Load the CSV file into a DataFrame
  df = pd.read_csv(
      file_path,
      header=None,
      skiprows=1,
      names=[
          "function", "precision", "x", "y", "z", "px", "py", "backend", "nodes", "jit_time",
          "min_time", "max_time", "mean_time", "std_div", "last_time", "generated_code",
          "argument_size", "output_size", "temp_size", "flops"
      ],
      dtype={
          "function": str,
          "precision": str,
          "x": int,
          "y": int,
          "z": int,
          "px": int,
          "py": int,
          "backend": str,
          "nodes": int,
          "jit_time": float,
          "min_time": float,
          "max_time": float,
          "mean_time": float,
          "std_div": float,
          "last_time": float,
          "generated_code": float,
          "argument_size": float,
          "output_size": float,
          "temp_size": float,
          "flops": float
      },
      index_col=False)
  # Group the data by FFT size (x, y, z) and compute the mean of the times
  df['size'] = df['x'] * df['y'] * df['z']
  df_grouped = df.groupby('size').mean()

  return df_grouped[['jit_time', 'min_time', 'max_time', 'mean_time', 'last_time']]


def compare_metrics(file_path1, file_path2):
  # Load and prepare the data from both files
  df1 = load_and_prepare_data(file_path1)
  df2 = load_and_prepare_data(file_path2)

  # Ensure both DataFrames have the same sizes
  common_sizes = df1.index.intersection(df2.index)
  df1 = df1.loc[common_sizes]
  df2 = df2.loc[common_sizes]

  # Calculate the percentage difference for each metric
  comparison = (df2 - df1) / df1 * 100

  return comparison


def main(file_path1, file_path2):
  comparison = compare_metrics(file_path1, file_path2)

  # Display the results
  print("Percentage difference between the two files:")
  print(comparison)


# Example usage
file_path1 = "v100/JAX.csv"
file_path2 = "v100/JAXDECOMP.csv"

main(file_path1, file_path2)
