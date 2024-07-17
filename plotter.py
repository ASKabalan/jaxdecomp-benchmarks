import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

np.seterr(divide='ignore')
plt.rcParams.update({'font.size': 10})
sns.plotting_context("paper")


def plot_by_gpus(dataframes,
                 simple_plot,
                 fixed_data_size,
                 nodes_in_label=False,
                 figure_size=(6, 4),
                 output=None,
                 dark_bg=False,
                 print_decompositions=False,
                 backends=None):

    if dark_bg:
        plt.style.use('dark_background')

    if backends is None:
        backends = ['MPI', 'NCCL', 'MPI4JAX']

    for method, df in dataframes.items():
        if not df[df['x'].isin(fixed_data_size)].empty:
            continue

    num_subplots = len(fixed_data_size)
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)
    if num_subplots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, data_size in enumerate(fixed_data_size):
        ax = axs[i]
        number_of_gpus = []
        times = []

        for method, df in dataframes.items():
            df = df[df['x'] == int(data_size)]
            if df.empty:
                continue
            df = df.sort_values(by=['gpus'])
            number_of_gpus.extend(df['gpus'].values)
            times.extend(df['time'].values)

            if simple_plot.get(method) is not None:
                label = f"{method}" if not nodes_in_label else f"{method}-{df['nodes'].values[0]}nodes"
                ax.plot(df['gpus'].values,
                        df['time'],
                        marker='o',
                        linestyle='-',
                        label=label)
                continue

            for backend in backends:
                df_backend = df[df['backend'] == backend]
                if df_backend.empty:
                    continue

                df_decomp = df_backend.groupby(['gpus', 'backend', 'nodes'])
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
                times.extend(sorted_df["time"].values)

                label = f"{method}-{backend}-{group['nodes'].values[0]}nodes" if nodes_in_label else f"{method}-{backend}"
                ax.plot(sorted_df['gpus'].values,
                        sorted_df['time'],
                        marker='o',
                        linestyle='-',
                        label=label)

                if print_decompositions:
                    for j, (px, py) in enumerate(
                            zip(sorted_df['px'], sorted_df['py'])):
                        ax.annotate(f"{px}x{py}",
                                    (sorted_df['gpus'].values[j],
                                     sorted_df['time'].values[j]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center',
                                    color='red' if j == 0 else 'white')

        ax.set_title(f"Data Size: {data_size}")
        unique_gpus = list(dict.fromkeys(number_of_gpus))
        unique_gpus.sort()

        if len(unique_gpus) == 0:
            fig.delaxes(ax)
            continue

        f2 = lambda a: np.log2(a)
        g2 = lambda b: b**2
        f10 = lambda a: np.log10(a)
        g10 = lambda b: b**2
        ax.set_xlim([min(unique_gpus), max(unique_gpus)])
        y_min, y_max = min(times) * 0.9, max(times) * 1.1
        ax.set_ylim([y_min, y_max])
        ax.set_xscale('function', functions=(f2, g2))
        ax.set_yscale('function', functions=(f10, g10))
        ax.set_xticks(unique_gpus)
        time_ticks = [
            10**t for t in range(int(np.floor(np.log10(y_min))), 1 +
                                 int(np.ceil(np.log10(y_max))))
        ]
        ax.set_yticks(time_ticks)

        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Time (secondes)')

        for x_value in unique_gpus:
            ax.axvline(x=x_value, color='gray', linestyle='--', alpha=0.5)

        ax.legend(loc='best')

    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    fig.tight_layout()
    rect = FancyBboxPatch((0.1, 0.1),
                          0.8,
                          0.8,
                          boxstyle="round,pad=0.02",
                          ec="black",
                          fc="none")
    fig.patches.append(rect)
    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight', transparent=False)


def plot_by_data_size(dataframes,
                      simple_plot,
                      fixed_gpu_size,
                      nodes_in_label=False,
                      figure_size=(6, 4),
                      output=None,
                      dark_bg=False,
                      print_decompositions=False,
                      backends=None):

    if dark_bg:
        plt.style.use('dark_background')

    plt.rcParams.update({'font.size': 10})

    if backends is None:
        backends = ['MPI', 'NCCL', 'MPI4JAX']

    for method, df in dataframes.items():
        if df[df['gpus'] == int(fixed_gpu_size[0])].empty:
            continue

    fixed_gpu_size = [
        gpu for gpu in fixed_gpu_size
        if not dataframes[list(dataframes.keys())[0]][dataframes[list(
            dataframes.keys())[0]]['gpus'] == int(gpu)].empty
    ]

    num_subplots = len(fixed_gpu_size)
    num_rows = int(np.ceil(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)

    if num_subplots > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, gpu_size in enumerate(fixed_gpu_size):
        ax = axs[i]
        data_sizes = []
        times = []

        for method, df in dataframes.items():
            df = df[df['gpus'] == int(gpu_size)]
            if df.empty:
                continue

            df = df.sort_values(by=['x'])
            data_sizes.extend(df['x'].values)
            times.extend(df['time'].values)

            if simple_plot.get(method) is not None:
                ax.plot(df['x'].values,
                        df['time'],
                        marker='o',
                        linestyle='-',
                        label=f"{method}")
                continue

            for backend in backends:
                df_backend = df[df['backend'] == backend]
                if df_backend.empty:
                    continue

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
                times.extend(sorted_df["time"].values)

                label = f"{method}-{backend}-{group['nodes'].values[0]}nodes" if nodes_in_label else f"{method}-{backend}"
                ax.plot(sorted_df['x'].values,
                        sorted_df['time'],
                        marker='o',
                        linestyle='-',
                        label=label)

                if print_decompositions:
                    for j, (px, py) in enumerate(
                            zip(sorted_df['px'], sorted_df['py'])):
                        ax.annotate(f"{px}x{py}",
                                    (sorted_df['x'].values[j],
                                     sorted_df['time'].values[j]),
                                    textcoords="offset points",
                                    xytext=(0, 10),
                                    ha='center',
                                    color='red' if j == 0 else 'white')

        ax.set_title(f"Number of GPUs: {gpu_size}")
        data_sizes = list(dict.fromkeys(data_sizes))
        data_sizes.sort()

        f2 = lambda a: np.log2(a)
        g2 = lambda b: b**2
        f10 = lambda a: np.log10(a)
        g10 = lambda b: b**2
        ax.set_xlim([min(data_sizes), max(data_sizes)])
        y_min, y_max = min(times) * 0.9, max(times) * 1.1
        ax.set_ylim([y_min, y_max])
        ax.set_xscale('function', functions=(f2, g2))
        ax.set_yscale('function', functions=(f10, g10))
        time_ticks = [
            10**t for t in range(int(np.floor(np.log10(y_min))), 1 +
                                 int(np.ceil(np.log10(y_max))))
        ]
        ax.set_xticks(data_sizes)
        ax.set_yticks(time_ticks)

        ax.set_xlabel('Data size (pixels³)')
        ax.set_ylabel('Time (secondes)')

        for x_value in data_sizes:
            ax.axvline(x=x_value, color='grey', linestyle=':', alpha=0.5)

        ax.legend(loc='lower right', fontsize='small')
        ax.tick_params(axis='y',
                       which='both',
                       labelleft=True,
                       labelright=False)
        ax.tick_params(axis='x')

    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    fig.tight_layout()
    rect = FancyBboxPatch((0.1, 0.1),
                          0.8,
                          0.8,
                          boxstyle="round,pad=0.02",
                          ec="black",
                          fc="none")
    fig.patches.append(rect)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches='tight', transparent=False, dpi=300)


def clean_up_csv(csv_files,
                 precision,
                 fft_type,
                 gpus=None,
                 data_sizes=None,
                 time_aggregation='mean'):
    dataframes = {}
    for csv_file in csv_files:
        file_name = os.path.splitext(os.path.basename(csv_file))[0]
        file_split = file_name.split("-")
        nodes = 1
        if len(file_split) > 1:
            file_name = file_split[0]
            nodes = int(file_split[1].split("node")[0])

        df = pd.read_csv(csv_file,
                         header=None,
                         names=[
                             "rank", "FFT_type", "precision", "x", "y", "z",
                             "px", "py", "backend", "nodes", "time"
                         ],
                         index_col=False)

        df = df[(df['precision'] == precision) & (df['FFT_type'] == fft_type)]

        if data_sizes:
            df = df[df['x'].isin(data_sizes)]

        grouped_df = df.groupby([
            "FFT_type", "precision", "x", "y", "z", "px", "py", "backend",
            "nodes"
        ])

        sub_dfs = [group for _, group in grouped_df]
        sub_dfs = [
            df.drop_duplicates([
                "rank", "FFT_type", "precision", "x", "y", "z", "px", "py",
                "backend", "nodes"
            ],
                               keep='last') for df in sub_dfs
        ]

        num_gpu = [len(sub_df) for sub_df in sub_dfs]
        num_node = [nodes for _ in sub_dfs]

        aggregated_dfs = []
        for sub_df in sub_dfs:
            if time_aggregation == 'mean':
                sub_df['time'] = sub_df['time'].mean()
            elif time_aggregation == 'min':
                sub_df['time'] = sub_df['time'].min()
            elif time_aggregation == 'max':
                sub_df['time'] = sub_df['time'].max()
            sub_df.drop(columns=['rank'], inplace=True)
            aggregated_dfs.append(sub_df.iloc[0])

        aggregated_df = pd.DataFrame(aggregated_dfs)
        aggregated_df['gpus'] = num_gpu

        if gpus:
            aggregated_df = aggregated_df[aggregated_df['gpus'].isin(gpus)]

        if dataframes.get(file_name) is None:
            dataframes[file_name] = aggregated_df
        else:
            dataframes[file_name] = pd.concat(
                [dataframes[file_name], aggregated_df])

    return dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot benchmark data')
    parser.add_argument('-f',
                        '--csv_files',
                        nargs='+',
                        help='List of csv files to plot')
    parser.add_argument('-g',
                        '--gpus',
                        nargs='*',
                        type=int,
                        help='List of number of gpus to plot')
    parser.add_argument('-d',
                        '--data_size',
                        nargs='*',
                        type=int,
                        help='List of data size to plot')
    parser.add_argument('-sc',
                        '--scaling',
                        choices=['Weak', 'Strong'],
                        required=True,
                        help='Scaling type (Weak or Strong)')
    parser.add_argument('-s',
                        '--simple_plot',
                        nargs='*',
                        help='List of methods to plot without backends')
    parser.add_argument('-fs',
                        '--figure_size',
                        nargs=2,
                        type=int,
                        help='Figure size')
    parser.add_argument('-nl',
                        '--nodes_in_label',
                        action='store_true',
                        help='Use node names in labels')
    parser.add_argument('-o', '--output', help='Output file', default=None)
    parser.add_argument('-ta',
                        '--time_aggregation',
                        choices=['mean', 'min', 'max'],
                        default='mean',
                        help='Time aggregation method')
    parser.add_argument('-db',
                        '--dark_bg',
                        type=bool,
                        default=False,
                        help='Use dark background for plotting')
    parser.add_argument('-pd',
                        '--print_decompositions',
                        action='store_true',
                        help='Print decompositions on plot')
    parser.add_argument('-b',
                        '--backends',
                        nargs='*',
                        default=['MPI', 'NCCL', 'MPI4JAX'],
                        help='List of backends to include')
    parser.add_argument('-p',
                        '--precision',
                        choices=['float32', 'float64'],
                        default='float32',
                        help='Precision to filter by (float32 or float64)')
    parser.add_argument('-t',
                        '--fft_type',
                        choices=['FFT', 'IFFT'],
                        default='FFT',
                        help='FFT type to filter by (FFT or IFFT)')

    args = parser.parse_args()

    simple_plot = {}
    if args.simple_plot is not None:
        for method in args.simple_plot:
            simple_plot[method] = True

    dataframes = clean_up_csv(args.csv_files, args.precision, args.fft_type,
                              args.gpus, args.data_size, args.time_aggregation)

    if args.scaling == 'Weak':
        plot_by_data_size(dataframes, simple_plot, args.gpus,
                          args.nodes_in_label, args.figure_size, args.output,
                          args.dark_bg, args.print_decompositions,
                          args.backends)
    elif args.scaling == 'Strong':
        plot_by_gpus(dataframes, simple_plot, args.data_size,
                     args.nodes_in_label, args.figure_size, args.output,
                     args.dark_bg, args.print_decompositions, args.backends)
