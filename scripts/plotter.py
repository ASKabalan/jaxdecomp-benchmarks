import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# To be used to plot the CSV comming from the benchmarks
# Usage example
# python plotter.py --csv_files JAX.csv JAXDECOMP.csv --gpus 4 8 16 --simple_plot JAX --figure_size 6 4 --nodes_in_label
# python plotter.py --csv_files JAX.csv JAXDECOMP.csv --data_size 128 256 512 1024 2048 --simple_plot JAX --figure_size 6 4 --nodes_in_label

# You should have one CSV per framework, each csv can contain results from different number of nodes
# for more info use python plotter.py -h


def plot_by_gpus(dataframes,
                 simple_plot,
                 fixed_data_size,
                 nodes_in_label=False,
                 figure_size=(6, 4)):

    # filter methods that do not have the data size
    for method, df in dataframes.items():
        if df[df['x'] == int(fixed_data_size[0])].empty:
            continue
    # filter data size not contained in the data
    fixed_data_size = [
        data_size for data_size in fixed_data_size
        if not dataframes[list(dataframes.keys())[0]][dataframes[list(
            dataframes.keys())[0]]['x'] == int(data_size)].empty
    ]

    # Calculate the number of subplots based on the number of data sizes
    num_subplots = len(fixed_data_size)
    # Calculate the number of rows and columns for the subplots
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create a figure and axis
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()  # Flatten the axs array for easier indexing
    else:
        axs = [axs]
    # Plot the data for each method
    for i, data_size in enumerate(fixed_data_size):
        # Select the current subplot
        ax = axs[i]
        number_of_gpus = []

        for method, df in dataframes.items():
            # filter for chosen data size
            df = df[df['x'] == int(data_size)]
            if df.empty:
                continue
            # sort by number of gpus
            df = df.sort_values(by=['gpus'])
            number_of_gpus.extend(df['gpus'].values)

            if simple_plot.get(method) is not None:
                label = f"{method}" if not nodes_in_label else f"{method}-{df['nodes'].values[0]}nodes"
                ax.plot(df['gpus'].values,
                        df['time'],
                        marker='o',
                        linestyle='-',
                        label=f"{method}")
                continue
            # filter by backend and plot one line for each backend
            for backend in 'MPI', 'NCCL':
                df_backend = df[df['backend'] == backend]

                if df_backend.empty:
                    continue

                # get the fastest decomposition (number of gpus and nodes must be fixed here)
                df_decomp = df_backend.groupby(
                    ['x', 'y', 'z', 'backend', 'nodes'])
                all_decomp = []
                sorted_dfs = []

                for _, group in df_decomp:
                    group.sort_values(by=['time'],
                                      inplace=True,
                                      ascending=False)
                    for px, py in zip(group['px'].values, group['py'].values):
                        all_decomp.append(f"{px}x{py}")
                    sorted_dfs.append(group.iloc[0])

                sorted_df = pd.DataFrame(sorted_dfs)

                label = f"{method}-{backend}-{group['nodes'].values[0]}nodes" if nodes_in_label else f"{method}-{backend}"

                ax.plot(sorted_df['gpus'].values,
                        sorted_df['time'],
                        marker='o',
                        linestyle='-',
                        label=label)

            # ad a decompostion text above the gpu number point [columns "px" and "py"]
            #for i, txt in enumerate(df.index):
            #  ax.annotate(txt, (df['gpus'].values[i], df['time'].values[i]), textcoords="offset points", xytext=(0,10), ha='center')

        # add title data size
        ax.set_title(f"Data Size: {data_size}")
        # Set x-axis ticks exactly at the provided data_size values
        ax.set_yscale('log')

        unique_gpus = list(dict.fromkeys(number_of_gpus))
        unique_gpus.sort()
        ax.set_xticks(list(unique_gpus))
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])

        # Set labels and title
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time')

        # Add a constant dotted line along the Y-axis at each X point
        for x_value in unique_gpus:
            ax.axvline(x=x_value, color='gray', linestyle='--', alpha=0.5)

        # Set y-axis ticks to be formatted as 10^(-values)
        ax.yaxis.set_major_formatter(
            ScalarFormatter(useMathText=True, useOffset=False))
        ax.tick_params(axis='y',
                       which='both',
                       labelleft=True,
                       labelright=False)

        # Add legend
        ax.legend()

    # Remove any unused subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Display the plot
    plt.show()


def plot_by_data_size(dataframes,
                      simple_plot,
                      fixed_gpu_size,
                      nodes_in_label=False,
                      figure_size=(6, 4)):

    # Filter methods that do not have the gpu size
    for method, df in dataframes.items():
        if df[df['gpus'] == int(fixed_gpu_size[0])].empty:
            continue
    # filter gpu size not contained in the data
    fixed_gpu_size = [
        gpu for gpu in fixed_gpu_size
        if not dataframes[list(dataframes.keys())[0]][dataframes[list(
            dataframes.keys())[0]]['gpus'] == int(gpu)].empty
    ]
    # Calculate the number of subplots based on the number of GPU sizes
    num_subplots = len(fixed_gpu_size)
    # Calculate the number of rows and columns for the subplots
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()  # Flatten the axs array for easier indexing
    else:
        axs = [axs]
    # Plot the data for each GPU size
    for i, gpu_size in enumerate(fixed_gpu_size):
        ax = axs[i]  # Select the current subplot

        data_sizes = []

        # Plot the data for each method
        for method, df in dataframes.items():
            # Filter for the chosen number of GPUs
            df = df[df['gpus'] == int(gpu_size)]
            if df.empty:
                continue
            # Sort by data size
            df = df.sort_values(by=['x'])
            data_sizes.extend(df['x'].values)

            # Filter by backend and plot one line for each backend
            if simple_plot.get(method) is not None:
                ax.plot(df['x'].values,
                        df['time'],
                        marker='o',
                        linestyle='-',
                        label=f"{method}")
                continue

            for backend in ['MPI', 'NCCL']:
                df_backend = df[df['backend'] == backend]

                if df_backend.empty:
                    continue

                # Get the fastest decomposition (number of GPUs and nodes must be fixed here)
                df_decomp = df_backend.groupby(
                    ['x', 'y', 'z', 'backend', 'nodes'])
                all_decomp = []
                sorted_dfs = []

                for _, group in df_decomp:
                    group.sort_values(by=['time'],
                                      inplace=True,
                                      ascending=False)
                    for px, py in zip(group['px'].values, group['py'].values):
                        all_decomp.append(f"{px}x{py}")
                    sorted_dfs.append(group.iloc[0])

                sorted_df = pd.DataFrame(sorted_dfs)
                label = f"{method}-{backend}-{group['nodes'].values[0]}nodes" if nodes_in_label else f"{method}-{backend}"

                ax.plot(sorted_df['x'].values,
                        sorted_df['time'],
                        marker='o',
                        linestyle='-',
                        label=label)

        # add title nb of gpus
        ax.set_title(f"Number of GPUs: {gpu_size}")
        # Set x-axis ticks exactly at the provided data size values
        ax.set_yscale('log')
        data_sizes = list(dict.fromkeys(data_sizes))
        data_sizes.sort()
        ax.set_xticks(data_sizes)
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1e0])

        # Set labels and title
        ax.set_xlabel('Data Size')
        ax.set_ylabel('Time')

        # Add a constant dotted line along the Y-axis at each X point
        for x_value in data_sizes:
            ax.axvline(x=x_value, color='gray', linestyle='--', alpha=0.5)

        # Set y-axis ticks to be formatted as 10^(-values)
        ax.yaxis.set_major_formatter(
            ScalarFormatter(useMathText=True, useOffset=False))
        ax.tick_params(axis='y',
                       which='both',
                       labelleft=True,
                       labelright=False)

        # Add legend
        ax.legend()

    # Remove any unused subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the plot
    plt.savefig("plot.png")


def clean_up_csv(csv_files):
    dataframes = {}
    for csv_file in csv_files:
        # Add a list with the base file name without extension to the dict
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        # file is jaxfft-1node.csv .. if we split by - we get jaxfft and 1node
        # set 1node as number of nodes in a collum if it exists other wise default value is 1
        file_split = file_name.split("-")
        nodes = 1
        if len(file_split) > 1:
            file_name = file_name[0]
            nodes = int(file_split[1].split("node")[0])

        # Get pandas dataframe from csv file
        df = pd.read_csv(csv_file,
                         header=None,
                         names=[
                             "rank", "x", "y", "z", "px", "py", "backend",
                             "nodes", "time"
                         ],
                         index_col=False)

        # Group by the specified columns
        grouped_df = df.groupby(
            ["x", "y", "z", "px", "py", "backend", "nodes"])

        # Create a list of DataFrames, each corresponding to a unique combination of columns
        sub_dfs = [group for _, group in grouped_df]

        sub_dfs = [
            df.drop_duplicates(
                ["rank", "x", "y", "z", "px", "py", "backend", "nodes"],
                keep='last') for df in sub_dfs
        ]

        # Add a new column for the number of elements (number of GPUs used) in each subgroup
        num_gpu = [len(sub_df) for sub_df in sub_dfs]
        num_node = [nodes for _ in sub_dfs]

        # Calculate the mean for each sub-DataFrame
        mean_dfs = []
        for sub_df in sub_dfs:
            sub_df['time'] = sub_df['time'].mean()
            sub_df.drop(columns=['rank'], inplace=True)
            # take the first row as the mean of the group
            mean_dfs.append(sub_df.iloc[0])

        # Create a new DataFrame with only the mean values
        mean_df = pd.DataFrame(mean_dfs)

        # Add the "num_elements" column before calculating the mean
        mean_df['gpus'] = num_gpu

        if dataframes.get(file_name) is None:
            dataframes[file_name] = mean_df
        else:
            dataframes[file_name] = pd.concat([dataframes[file_name], mean_df])

    return dataframes


if __name__ == "__main__":

    # Get a list of csv files to plot from argv using argparser
    parser = argparse.ArgumentParser(description='Plot benchmark data')
    parser.add_argument('-f',
                        '--csv_files',
                        nargs='+',
                        help='List of csv files to plot')
    # chose and fix number of gpu for plot [this is a list]
    parser.add_argument('-g',
                        '--gpus',
                        nargs='*',
                        help='List of number of gpus to plot')
    # chose and fix data size for plot
    parser.add_argument('-d',
                        '--data_size',
                        nargs='*',
                        help='List of data size to plot')
    # Plot without backs for these methods
    parser.add_argument('-s',
                        '--simple_plot',
                        nargs='*',
                        help='List of methods to plot without backends')
    # figure size
    parser.add_argument('-fs',
                        '--figure_size',
                        nargs=2,
                        type=int,
                        help='Figure size')
    # use node names in labels
    parser.add_argument('-nl',
                        '--nodes_in_label',
                        action='store_true',
                        help='Use node names in labels')

    args = parser.parse_args()

    simple_plot = {}
    if args.simple_plot is not None:
        for method in args.simple_plot:
            simple_plot[method] = True
    # Clean up the csv files
    dataframes = clean_up_csv(args.csv_files)

    if args.gpus is not None and args.data_size is not None:
        print("You must choose at least one of the options -g or -d")
        sys.exit(1)

    if args.gpus is not None:
        plot_by_data_size(dataframes, simple_plot, args.gpus,
                          args.nodes_in_label, args.figure_size)
    if args.data_size is not None:
        plot_by_gpus(dataframes, simple_plot, args.data_size,
                     args.nodes_in_label, args.figure_size)
