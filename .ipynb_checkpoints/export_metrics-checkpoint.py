import pdb
from aim import Repo
import numpy as np
import pandas as pd

repo = Repo("./")
query = 'metric.name == "loss" and run.experiment == "CAMEL CASE" and run.h_params.mode in ["UTCGP-NUM", "MTCGP-NUM", "UTCGP-NUM-NOPRECOR"]'

print(repo)

steps = list(range(25_000))
Vs = []
# Get collection of metrics
for run_metrics_collection in repo.query_metrics(query).iter_runs():
    for metric in run_metrics_collection:
        v = np.zeros(25_000)
        exp = metric.run.experiment
        hash = metric.run.hash
        mode = metric.run["h_params"]["mode"]
        # Get run params
        params = metric.run[...]
        print(params)
        # Get metric values
        steps, metric_values = metric.values.sparse_numpy()
        n = metric_values.shape[0]
        v[:n] = metric_values
        data_pack = {"hash": hash, "mode": mode, "values": v}
        Vs.append(data_pack)
        break
    break
