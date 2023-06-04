from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import r2_score
from tqdm import tqdm

from mlpf.data.data.torch.power_flow import power_flow_data, get_relative_power_flow_errors
from mlpf.data.loading.load_data import autodetect_load_ppc
from mlpf.utils.standard_scaler import StandardScalar


def list_of_dicts2dict_of_lists(list_of_dicts: List[Dict]) -> Dict:
    return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}


def collect_logs(loss, predictions, batch):
    # TODO dict could be a dataclass

    # TODO the averaging is gonna make a biased estimate when the last batch is a lot smaller than the rest; maybe saving every sample and not batching would be ok?

    relative_active_power_errors, relative_reactive_power_errors = get_relative_power_flow_errors(predictions.detach().cpu(), batch)
    r2score = r2_score(predictions.detach().cpu(), batch.target_vector, multioutput="raw_values").detach()
    r2score[torch.isinf(r2score)] = 0.0

    return {
        "loss": loss.item(),
        "r2_score": torch.mean(r2score).item(),
        "rel P error mean": torch.mean(relative_active_power_errors).item(),
        "rel P error median": torch.median(relative_active_power_errors).item()
    }


def main():
    # Random seeds
    pyg.seed_everything(123)

    # Hyperparameters
    num_epochs = 1000
    batch_size = 512
    learning_rate = 3e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ppcs and ppc -> Data

    ppc_list = autodetect_load_ppc("generated_ppcs", shuffle=True, max_samples=1000)

    pf_data_list = []
    for solved_ppc in tqdm(ppc_list, ascii=True, desc="Converting ppcs to data"):
        pf_data_list.append(power_flow_data(solved_ppc))

    data_train, data_val = train_test_split(pf_data_list, test_size=0.33, random_state=42)

    # Torch dataloaders

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    input_size = data_train[0].feature_vector.shape[1]
    output_size = data_train[0].target_vector.shape[1]

    train_features = torch.vstack([data.feature_vector for data in data_train])

    # Model
    model = nn.Sequential(
        StandardScalar(train_features),
        nn.Linear(in_features=input_size, out_features=output_size),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    progress_bar = tqdm(range(num_epochs), ascii=True, desc="Training | Validation:")

    for epoch in progress_bar:
        train_logs = []
        val_logs = []

        # Training
        model.train()
        for batch in train_loader:
            features, targets = batch.feature_vector, batch.target_vector
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()

            train_logs.append(collect_logs(loss, predictions, batch))

            optimizer.step()

        # Validation
        with torch.no_grad():

            model.eval()
            for batch in val_loader:
                features, targets = batch.feature_vector, batch.target_vector
                features = features.to(device)
                targets = targets.to(device)

                predictions = model(features)
                loss = criterion(predictions, targets)

                val_logs.append(collect_logs(loss, predictions, batch))

        # Logging
        train_log = list_of_dicts2dict_of_lists(train_logs)
        val_log = list_of_dicts2dict_of_lists(val_logs)

        loss_str = f"loss: ({np.mean(train_log['loss']):2.4f} | {np.mean(val_log['loss']):2.4f})"
        r2score_str = f"r2score: ({np.mean(train_log['r2_score']):2.4f} | {np.mean(val_log['r2_score']):2.4f})"
        mean_rel_P_error_str = f"mean: ({np.mean(train_log['rel P error mean']):2.4f} | {np.mean(val_log['rel P error mean']):2.4f})"
        median_rel_P_error_str = f"median: ({np.mean(train_log['rel P error median']):2.4f} | {np.mean(val_log['rel P error median']):2.4f})"

        # the PF errors reach a minimum but then start climbing again while the loss keeps declining because the loss focuses on minimizing some larger scale
        # components of itself while neglecting the smaller ones which has a noticeable effect on the PF.
        progress_bar.set_description(
            f"Training | Validation: {loss_str}; {r2score_str}; relative active power error: {mean_rel_P_error_str}, {median_rel_P_error_str}"
        )


if __name__ == '__main__':
    main()
