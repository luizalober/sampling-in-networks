# Prevents underlying C-libraries from multiplying threads inside parallel processes
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import os, glob, re

from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm

from .metrics import *
from .sampling import *
from .statistics import *


def _sample_worker(args):
    idx, graph, sampling_method, size = args
    sample = sampling_method(graph, size)
    return {
        'graph_id': idx,
        'sampling_method': sampling_method.__name__,
        'sample': sample,
        'sample_size': size,
    }

def _single_metric_worker(args):
    """Executes a single metric function on one sample graph."""
    graph, function, n_moments = args
    return function(graph, n_moments=n_moments)


class Samnet_ochestrator:

    def __init__(self, graph_list):
        self.graph_list = graph_list
        self.samples = []
        self.sample_metrics = []
        self.metrics_df = None

    def _get_default_workers(self, max_workers):
        """Leaves at least 1 core free for the OS to prevent freezing."""
        if max_workers is None:
            total_cores = os.cpu_count() or 1
            return max(1, total_cores - 1)
        return max_workers

    def sample_genesis(
        self, sampling_methods_list, sample_sizes, runs=1, max_workers=None
    ):
        samples = []

        # Baseline original graphs
        for idx, graph in enumerate(self.graph_list):
            samples.append({
                'graph_id': idx,
                'sampling_method': 'Original',
                'sample': graph,
                'sample_size': len(graph),
            })

        tasks = [
            (idx, graph, method, size)
            for idx, graph in enumerate(self.graph_list)
            for method in sampling_methods_list
            for size in sample_sizes
            for _ in range(runs)
        ]

        workers = self._get_default_workers(max_workers)

        if workers == 1:
            for task in tqdm(tasks, desc='Sampling graphs'):
                samples.append(_sample_worker(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                chunksize = max(1, len(tasks) // (workers))
                results = list(
                    tqdm(
                        executor.map(
                            _sample_worker, tasks, chunksize=chunksize
                        ),
                        total=len(tasks),
                        desc='Sampling graphs',
                    )
                )
                samples.extend(results)

        self.samples = samples

    def sample_describer(
        self,
        functions_list,
        n_moments=1,
        max_workers=None,
        checkpoint_dir=None,
    ):
        workers = self._get_default_workers(max_workers)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        #Precompute sample metadata
        base_metadata = [
            {
                'graph_id': sample_instance['graph_id'],
                'sampling_method': sample_instance['sampling_method'],
                'sample_size': sample_instance['sample_size'],
                'sample_edges': len(sample_instance['sample'].edges),
                'net_density': nx.density(sample_instance['sample']),
            }
            for sample_instance in self.samples
        ]
        self.sample_metrics = [meta.copy() for meta in base_metadata]

        for stage_idx, function in enumerate(functions_list, start=1):
            func_name = getattr(function, '__name__', f'metric_{stage_idx}')
            tasks = [
                (sample_instance['sample'], function, n_moments)
                for sample_instance in self.samples
            ]

            desc = f'Metric {stage_idx}/{len(functions_list)} ({func_name})'

            if workers == 1:
                results = [
                    _single_metric_worker(task)
                    for task in tqdm(tasks, desc=desc)
                ]
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    chunksize = max(1, len(tasks) // (workers * 4))
                    results = list(
                        tqdm(
                            executor.map(
                                _single_metric_worker,
                                tasks,
                                chunksize=chunksize,
                            ),
                            total=len(tasks),
                            desc=desc,
                        )
                    )

            stage_records = []
            for idx, (meta, res) in enumerate(zip(base_metadata, results)):
                stage_rec = meta.copy()

                if isinstance(res, dict):
                    for sub_key, val in res.items():
                        col_name = f"{func_name}_{sub_key}"
                        self.sample_metrics[idx][col_name] = val
                        stage_rec[col_name] = val
                else:
                    self.sample_metrics[idx][func_name] = res
                    stage_rec[func_name] = res

                stage_records.append(stage_rec)

            if checkpoint_dir:
                checkpoint_filename = f'stage_{stage_idx:02d}_{func_name}.csv'
                stage_df = pd.DataFrame(stage_records)
                stage_df.to_csv(
                    os.path.join(checkpoint_dir, checkpoint_filename),
                    index=False,
                )

        self.metrics_df = pd.DataFrame(self.sample_metrics)
        return self.metrics_df

    def combine_stage_checkpoints(
        self, checkpoint_dir='data/', output_filename='combined_metrics.csv'
    ):
        """Reads stage_{n}_{metric}.csv files, combines them side-by-side

        without memory explosion, and updates self.metrics_df.
        """
        pattern = os.path.join(checkpoint_dir, 'stage_*_*.csv')
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(
                f"No stage checkpoint files found in '{checkpoint_dir}'."
            )

        def extract_stage_num(filepath):
            match = re.search(r'stage_(\d+)_', os.path.basename(filepath))
            return int(match.group(1)) if match else 0

        files.sort(key=extract_stage_num)
        metadata_cols = ['graph_id', 'sampling_method', 'sample_size', 'sample_edges', 'net_density']

        base_df = pd.read_csv(files[0])
        dfs_to_concat = [base_df]

        for filepath in files[1:]:
            df = pd.read_csv(filepath)
            row_match = (
                (df[metadata_cols] == base_df[metadata_cols]).all().all()
            )
            if not row_match:
                raise ValueError(
                    f"Row mismatch detected in file '{os.path.basename(filepath)}'."
                )

            metric_cols = [c for c in df.columns if c not in metadata_cols]
            dfs_to_concat.append(df[metric_cols])

        self.metrics_df = pd.concat(dfs_to_concat, axis=1)

        if output_filename:
            output_path = os.path.join(checkpoint_dir, output_filename)
            self.metrics_df.to_csv(output_path, index=False)
            print(
                f"Successfully combined {len(files)} files into '{output_path}' "
                f"({len(self.metrics_df)} rows, {len(self.metrics_df.columns)} columns)."
            )

        return self.metrics_df

    def save_metrics_file(self, path=None, save=False):
        if self.metrics_df is None:
            self.metrics_df = pd.DataFrame(self.sample_metrics)

        if save and path:
            self.metrics_df.to_csv(path, index=False)
            return
        return self.metrics_df
