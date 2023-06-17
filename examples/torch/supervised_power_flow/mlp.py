import torch

import torch.nn as nn
import torch_geometric as pyg

from pandas.io.json._normalize import nested_to_record
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torchmetrics import MetricCollection, MeanSquaredError, R2Score
from tqdm import tqdm

from mlpf.data.data.torch.power_flow import power_flow_data
from mlpf.data.loading.load_data import autodetect_load_ppc
from mlpf.loss.torch.metrics.power_flow import RelativePowerFlowError
from mlpf.utils.standard_scaler import StandardScaler


def main():
    # Random seeds
    pyg.seed_everything(123)

    # Hyperparameters
    num_epochs = 1000
    batch_size = 64
    learning_rate = 3e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ppcs and ppc -> Data

    solved_ppc_list = autodetect_load_ppc("generated_ppcs", shuffle=True, max_samples=1000)

    pf_data_list = []
    for solved_ppc in tqdm(solved_ppc_list, ascii=True, desc="Converting ppcs to data"):
        pf_data_list.append(power_flow_data(solved_ppc))

    data_train, data_val = train_test_split(pf_data_list, test_size=0.33, random_state=42)

    # Torch dataloaders

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    input_size = data_train[0].feature_vector.shape[1]
    output_size = data_train[0].target_vector.shape[1]

    train_features = torch.vstack([data.feature_vector for data in data_train])
    train_targets = torch.vstack([data.target_vector for data in data_train])

    # the output variables are of varying orders of magnitude so not normalizing them will the MSE loss
    # favoring larger variables and the weights in the final layer being of varying orders of magnitude themselves.
    # when calculating the power flow loss, an inverse transform needs to be applied to the predictions
    output_scaler = StandardScaler(train_targets)
    output_scaler.to(device)

    # Model
    model = nn.Sequential(
        StandardScaler(train_features),
        nn.Linear(in_features=input_size, out_features=output_size),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    metrics_train = MetricCollection(MeanSquaredError(), R2Score(num_outputs=output_size), RelativePowerFlowError()).to(device)
    metrics_val = MetricCollection(MeanSquaredError(), R2Score(num_outputs=output_size), RelativePowerFlowError()).to(device)
    # metrics_train = MetricCollection(MeanSquaredError(), R2Score(num_outputs=output_size)).to(device)
    # metrics_val = MetricCollection(MeanSquaredError(), R2Score(num_outputs=output_size)).to(device)

    progress_bar = tqdm(range(num_epochs), ascii=True, desc="Training | Validation:")

    for epoch in progress_bar:

        # Training
        model.train()
        for batch in train_loader:
            features, targets = batch.feature_vector, batch.target_vector
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(features)
            loss = criterion(predictions, output_scaler(targets))
            loss.backward()

            metrics_train(preds=predictions, target=output_scaler(targets), preds_pf=output_scaler.inverse(predictions), batch=batch)

            optimizer.step()

        # Validation
        with torch.no_grad():

            model.eval()
            for batch in val_loader:
                features, targets = batch.feature_vector, batch.target_vector
                features = features.to(device)
                targets = targets.to(device)

                predictions = model(features)

                metrics_val(preds=predictions, target=output_scaler(targets), preds_pf=output_scaler.inverse(predictions), batch=batch)

        overall_metrics_train = nested_to_record(metrics_train.compute(), sep='_')
        overall_metrics_val = nested_to_record(metrics_val.compute(), sep='_')

        description = "Training | Validation:"

        for key in overall_metrics_train.keys():
            description += f" {key}: ({overall_metrics_train[key]:2.4f} | {overall_metrics_val[key]:2.4f});"

        progress_bar.set_description(description)

        metrics_train.reset()
        metrics_val.reset()


if __name__ == '__main__':
    main()
