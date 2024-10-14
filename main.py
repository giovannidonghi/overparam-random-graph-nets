import os
import time
import argparse
import itertools
import numpy as np
import pandas as pd
from random import seed

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.preprocessing import StandardScaler

from models import  transform_with_untrained_GCN, UNTRAINEDGCN
from utils import train_rocket_fixed_alpha, printParOnFile, split_ids, X_by_layer, add_dummy_labels, renyi_entropy, random_subarray


def main(verbose, dataset_name, n_folds, n_run, max_iter, normalization, row_fraction):

    dataset_path = "dataset"

    #GRID parameter sets
    theta_list = [0.01, 0.1, 0.5, 1, 1.66, 3, 5, 10, 30, 50]
    n_unit_list = [10, 20, 30, 50, 75, 100, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000]
    alpha_list = [0] + np.logspace(-4, 5, 10).tolist()
    batch_size = 100
    act = 'TANH'
    out = 'ROCKET'
    optim = 'ridgeclassifier'
    max_k = 1
    n_layers = 4
    rnd_state = np.random.RandomState(seed(1))

    test_type = f"{out}_GCN"
    training_log_dir = os.path.join("test_log", test_type)
    if not os.path.exists(training_log_dir):
        os.makedirs(training_log_dir)
    results_dir = os.path.join("results", test_type)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    processed_dir = os.path.join("processed", dataset_name)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    index_combinations = list(itertools.product(theta_list, n_unit_list, range(n_run), range(n_folds), alpha_list, range(5)))
    multiindex = pd.MultiIndex.from_tuples(index_combinations, names=['theta', 'n_units', 'run', 'fold', 'alpha', 'layer'])
    results_df = pd.DataFrame(index=multiindex, columns=["train", "val", "test"])
    results_df = results_df.astype(np.float64)

    index_combinations = list(itertools.product(theta_list, n_unit_list, range(n_run), range(n_folds), range(5)))
    multiindex = pd.MultiIndex.from_tuples(index_combinations, names=['theta', 'n_units', 'run', 'fold', 'layer'])
    metrics_df = pd.DataFrame(index=multiindex, columns=["s_max", "s_min", "entropy_m", "entropy_sd"])
    metrics_df = metrics_df.astype(np.float64)

    pd.set_option('display.max_columns', 20)

    device = torch.device('cpu')

    dataset = TUDataset(root=dataset_path, name=dataset_name, transform=T.SIGN(max_k), pre_transform=add_dummy_labels, use_node_attr=True)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True)

    start_time = time.time()

    for n_units in n_unit_list:
        for theta in theta_list:
            print(f"\nN units {n_units} \t Theta {theta}")
            runs_summaries = []
            for run in range(n_run):
                print(f"RUN {run}", flush=True)

                test_name = f"data-{dataset_name}_nHidden-{n_units}_theta-{theta}_run-{run}_nFold-{n_folds}_batchSize-{batch_size}_maxK-{max_k}_out-{out}_norm-{normalization}_act-{act}_optim-{optim}"

                printParOnFile(test_name=test_name, log_dir=training_log_dir,
                                                    par_list={"dataset_name": dataset_name,
                                                                "n_fold": n_folds,
                                                                "batch_size": batch_size,
                                                                "n_hidden": n_units,
                                                                "theta": theta,
                                                                "out":out,
                                                                "norm":normalization,
                                                                "act":act,
                                                                "optim":optim,
                                                                "max_k": max_k,})

                transformed_path = os.path.join(processed_dir, f"tetha-{theta}_units-{n_units}_run-{run}.npz")
                if os.path.exists(transformed_path):
                    transformed_data = np.load(transformed_path)
                    X_transform = transformed_data["X_transform"]
                    y = transformed_data["y"]
                else:
                    rocketgcn = UNTRAINEDGCN(n_feat=loader.dataset.num_features, hidden_dim=n_units,out=out,act=act, theta= theta).to(device)

                    X_transform, y = transform_with_untrained_GCN(rocketgcn, loader, device=device)
                    np.savez_compressed(os.path.join(processed_dir, f"theta-{theta}_units-{n_units}_run-{run}.npz"), X_transform=X_transform, y=y)
                    y = y.ravel()

                train_ids, test_ids, valid_ids = split_ids(rnd_state.permutation(len(dataset)), folds=n_folds)

                for fold_id in range(n_folds):
                    X_train_transform, y_train = X_transform[train_ids[fold_id]], y[train_ids[fold_id]]
                    X_valid_transform, y_valid = X_transform[valid_ids[fold_id]], y[valid_ids[fold_id]]
                    X_test_transform, y_test = X_transform[test_ids[fold_id]], y[test_ids[fold_id]]
                    
                    if normalization:
                        scaler = StandardScaler()
                        X_train_transform=scaler.fit_transform(X_train_transform)
                        X_valid_transform=scaler.transform(X_valid_transform)
                        X_test_transform=scaler.transform(X_test_transform)

                    # CONDITIONING
                    G = np.matmul(X_train_transform, X_train_transform.T)
                    svals = np.linalg.svd(G, compute_uv=False)
                    s_max, s_min = svals[0], svals[-1]

                    metrics_df.loc[theta, n_units, run, fold_id, 0] = (s_max, s_min, np.nan, np.nan)

                    # TRAINING
                    for alpha in alpha_list:
                        train_acc, test_acc, val_acc, _ = (train_rocket_fixed_alpha(X_train_transform, X_valid_transform, X_test_transform, y_train, y_valid, y_test, alpha, max_iter, optim))
                        results_df.loc[theta, n_units, run, fold_id, alpha, 0] = (train_acc, val_acc, test_acc)

                    X_train_layer = X_by_layer(X_train_transform, n_layers)
                    X_valid_layer = X_by_layer(X_valid_transform, n_layers)
                    X_test_layer = X_by_layer(X_test_transform, n_layers)
                    for l in range(n_layers):
                        # CONDITIONING
                        G = np.matmul(X_train_layer[l], X_train_layer[l].T)
                        svals = np.linalg.svd(G, compute_uv=False)
                        s_max, s_min = svals[0], svals[-1]
                        # ENTROPY
                        subsample = random_subarray(X_train_layer[l], row_fraction=row_fraction, max_columns=1000)
                        entropy_m, entropy_sd = renyi_entropy(subsample)

                        metrics_df.loc[theta, n_units, run, fold_id, l+1] = (s_max, s_min, entropy_m, entropy_sd)
                        
                        # TRAINING
                        for alpha in alpha_list:
                            train_acc, test_acc, val_acc, _ = (train_rocket_fixed_alpha(X_train_layer[l], X_valid_layer[l], X_test_layer[l], y_train, y_valid, y_test, alpha, max_iter, optim))
                            results_df.loc[theta, n_units, run, fold_id, alpha, l+1] = (train_acc, val_acc, test_acc)

                filtered_df = results_df.xs((theta, n_units, run, 0), level=('theta', 'n_units', 'run', 'layer'))
                max_alpha_rows = filtered_df.loc[filtered_df.groupby(level='fold')['val'].idxmax()]
                results_summary = max_alpha_rows.agg(['mean', 'std'])
                metrics_summary = metrics_df.xs((theta, n_units, run, 0), level=('theta', 'n_units', 'run', 'layer')).agg(['mean', 'std'])
                summary = results_summary.join(metrics_summary)
                runs_summaries.append(summary)
                if verbose:
                    print(summary)
                    print(f"Partial time run {run}: {time.time() - start_time:.2f}s")
                with open(os.path.join(training_log_dir,test_name+".log"),'a') as log:
                    log.write("\n" + summary.to_string())
                
            #stats over the runs
            print(f"Theta {theta} \t N units {n_units} \t N_examples {X_transform.shape[0]}")
            merged_df = pd.concat(runs_summaries).loc["mean"]
            print(merged_df.agg(['mean', 'std']).to_string(index=False))

            print(f"--- {time.time() - start_time:.2f} seconds ---")

    results_df.to_csv(os.path.join(results_dir, f"{dataset_name}_results.csv"))
    metrics_df.to_csv(os.path.join(results_dir, f"{dataset_name}_metrics.csv"))
    print('Results saved successfully to file')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='PTC_MR', help="PTC_MR, DD, ENZYMES, IMDB-BINARY, IMDB-MULTI, NCI1, PROTEINS")
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--n_run', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=2000)
    parser.add_argument('--normalization', type=bool, default=True)
    parser.add_argument('--row_fraction', type=float, default=0.1, help='fraction of graphs to sample for entropy calculation')

    args = parser.parse_args()

    main(args.verbose, args.dataset_name, args.n_folds, args.n_run, args.max_iter, args.normalization, args.fraction)
