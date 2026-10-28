import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import os, glob, re
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import networkx as nx
from tqdm import tqdm

from .metrics import *
from .sampling import *
from .statistics import *


_GLOBAL_GRAPHS = []

def _init_worker(graphs):
    """Initializes worker process with global graph references once per worker."""
    global _GLOBAL_GRAPHS
    _GLOBAL_GRAPHS = graphs

def _sample_worker(args):
    """Executes sampling worker returning lightweight node set IDs."""
    graph_idx, sampling_method, size = args
    graph = _GLOBAL_GRAPHS[graph_idx]
    sample = sampling_method(graph, size)

    return {
        'graph_id': graph_idx,
        'sampling_method': sampling_method.__name__,
        'sample_nodes': set(sample.nodes()),  #Storing node IDs set
        'sample_size': size,
    }

def _single_metric_worker(args):
    """Induces subgraph on-the-fly and evaluates a single metric function."""
    sample_idx, graph_id, sample_nodes, function, n_moments = args
    graph = _GLOBAL_GRAPHS[graph_id]
    subgraph = graph.subgraph(sample_nodes)
    return sample_idx, function(subgraph, n_moments=n_moments)


class Samnet_ochestrator:

    def __init__(self, graph_list):
        self.graph_list = graph_list
        self.samples = []
        self.sample_metrics = []
        self.metrics_df = None

    def _get_default_workers(self, max_workers):
        if max_workers is None:
            total_cores = os.cpu_count() or 1
            return max(1, total_cores - 1)
        return max_workers

    def sample_genesis(
        self, sampling_methods_list, sample_sizes, runs=1, max_workers=None
    ):
        samples = []

        #Original graphs
        for idx, graph in enumerate(self.graph_list):
            samples.append({
                'graph_id': idx,
                'sampling_method': 'Original',
                'sample_nodes': set(graph.nodes()),
                'sample_size': len(graph),
            })

        tasks = [
            (idx, method, size)
            for idx, _ in enumerate(self.graph_list)
            for method in sampling_methods_list
            for size in sample_sizes
            for _ in range(runs)
        ]

        workers = self._get_default_workers(max_workers)

        if workers == 1:
            _init_worker(self.graph_list)
            for task in tqdm(tasks, desc='Sampling graphs'):
                samples.append(_sample_worker(task))
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(self.graph_list,),
            ) as executor:
                chunksize = max(1, len(tasks) // (workers * 4))
                results = list(
                    tqdm(
                        executor.map(_sample_worker, tasks, chunksize=chunksize),
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
        save_frequency=10,  #Save temp checkpoint every N samples
    ):
        from concurrent.futures import as_completed

        workers = self._get_default_workers(max_workers)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        #Compute metadata with on-the-fly induced subgraphs
        base_metadata = []
        for sample_instance in self.samples:
            graph_id = sample_instance['graph_id']
            sample_nodes = sample_instance['sample_nodes']
            subgraph = self.graph_list[graph_id].subgraph(sample_nodes)

            base_metadata.append({
                'graph_id': graph_id,
                'sampling_method': sample_instance['sampling_method'],
                'sample_size': sample_instance['sample_size'],
                'sample_edges': subgraph.number_of_edges(),
                'net_density': nx.density(subgraph),
            })

        self.sample_metrics = [meta.copy() for meta in base_metadata]

        for stage_idx, function in enumerate(functions_list, start=1):
            func_name = getattr(function, '__name__', f'metric_{stage_idx}')
            final_checkpoint = (
                os.path.join(checkpoint_dir, f'stage_{stage_idx:02d}_{func_name}.csv')
                if checkpoint_dir else None
            )
            temp_checkpoint = (
                os.path.join(checkpoint_dir, f'temp_stage_{stage_idx:02d}_{func_name}.csv')
                if checkpoint_dir else None
            )

            # --- 1. FULL STAGE SKIP CHECK ---
            if final_checkpoint and os.path.exists(final_checkpoint):
                print(f"Stage {stage_idx}/{len(functions_list)} ({func_name}) already finished. Loading checkpoint...")
                stage_df = pd.read_csv(final_checkpoint)
                metric_cols = [c for c in stage_df.columns if c not in base_metadata[0].keys()]
                for idx, row in stage_df.iterrows():
                    for col in metric_cols:
                        self.sample_metrics[idx][col] = row[col]
                continue

            # --- 2. INTRA-STAGE TEMP RECOVERY CHECK ---
            stage_records_map = {}
            completed_indices = set()

            if temp_checkpoint and os.path.exists(temp_checkpoint):
                temp_df = pd.read_csv(temp_checkpoint)
                if 'sample_idx' in temp_df.columns:
                    completed_indices = set(temp_df['sample_idx'].tolist())
                    for _, row in temp_df.iterrows():
                        idx = int(row['sample_idx'])
                        record_dict = row.to_dict()
                        stage_records_map[idx] = record_dict

                        #Populate self.sample_metrics from temp file
                        for col, val in record_dict.items():
                            if col not in base_metadata[0].keys() and col != 'sample_idx':
                                self.sample_metrics[idx][col] = val

                    print(f"Resuming stage {stage_idx} ({func_name}): {len(completed_indices)}/{len(self.samples)} samples restored from temp file.")

            #Filter remaining tasks
            all_tasks = [
                (
                    idx,
                    sample_instance['graph_id'],
                    sample_instance['sample_nodes'],
                    function,
                    n_moments,
                )
                for idx, sample_instance in enumerate(self.samples)
            ]
            pending_tasks = [t for t in all_tasks if t[0] not in completed_indices]

            desc = f'Metric {stage_idx}/{len(functions_list)} ({func_name})'

            if pending_tasks:
                processed_count = 0

                def process_result(sample_idx, res):
                    meta = base_metadata[sample_idx]
                    stage_rec = meta.copy()
                    stage_rec['sample_idx'] = sample_idx

                    if isinstance(res, dict):
                        for sub_key, val in res.items():
                            col_name = f"{func_name}_{sub_key}"
                            self.sample_metrics[sample_idx][col_name] = val
                            stage_rec[col_name] = val
                    else:
                        self.sample_metrics[sample_idx][func_name] = res
                        stage_rec[func_name] = res

                    stage_records_map[sample_idx] = stage_rec

                if workers == 1:
                    _init_worker(self.graph_list)
                    for task in tqdm(pending_tasks, desc=desc):
                        s_idx, res = _single_metric_worker(task)
                        process_result(s_idx, res)
                        processed_count += 1

                        if checkpoint_dir and (processed_count % save_frequency == 0):
                            temp_df = pd.DataFrame(list(stage_records_map.values()))
                            temp_df.to_csv(temp_checkpoint, index=False)
                else:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_init_worker,
                        initargs=(self.graph_list,),
                    ) as executor:
                        futures = [executor.submit(_single_metric_worker, task) for task in pending_tasks]
                        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                            s_idx, res = future.result()
                            process_result(s_idx, res)
                            processed_count += 1

                            if checkpoint_dir and (processed_count % save_frequency == 0):
                                temp_df = pd.DataFrame(list(stage_records_map.values()))
                                temp_df.to_csv(temp_checkpoint, index=False)

            # --- 3. FINAL STAGE SAVE & TEMP CLEANUP ---
            if checkpoint_dir:
                #Order records by original sample index
                final_records = [stage_records_map[i] for i in range(len(self.samples))]
                stage_df = pd.DataFrame(final_records)
                if 'sample_idx' in stage_df.columns:
                    stage_df = stage_df.drop(columns=['sample_idx'])

                stage_df.to_csv(final_checkpoint, index=False)

                #Remove temp file upon stage completion
                if temp_checkpoint and os.path.exists(temp_checkpoint):
                    os.remove(temp_checkpoint)

        self.metrics_df = pd.DataFrame(self.sample_metrics)
        return self.metrics_df

    def sample_and_describe_pipeline(
        self,
        sampling_methods_list,
        sample_sizes,
        functions_list,
        runs=1,
        n_moments=1,
        max_workers=None,
        checkpoint_dir=None,
    ):
        self.sample_genesis(
            sampling_methods_list=sampling_methods_list,
            sample_sizes=sample_sizes,
            runs=runs,
            max_workers=max_workers,
        )

        df_results = self.sample_describer(
            functions_list=functions_list,
            n_moments=n_moments,
            max_workers=max_workers,
            checkpoint_dir=checkpoint_dir,
        )

        return df_results

    def combine_stage_checkpoints(
        self, checkpoint_dir='data/', output_filename='combined_metrics.csv'
    ):
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
