import torch
import torch.nn as nn
import torch_geometric as pyg
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from examples.torch.supervised_power_flow.utils import supervised_pf_logs, logs2str
from mlpf.data.data.torch.power_flow import power_flow_data
from mlpf.data.loading.load_data import autodetect_load_ppc
from mlpf.utils.standard_scaler import StandardScaler


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, standard_scaler):
        super(GNN, self).__init__()
        self.standard_scaler = standard_scaler
        self.graph_encoder = pyg.nn.GCN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=hidden_channels)
        self.linear = nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, data):
        out = self.standard_scaler(data.x)
        out = self.graph_encoder(x=out, edge_index=data.edge_index)
        out = self.linear(out)[~data.feature_mask].reshape(data.target_vector.shape)

        return out


def main():
    # TODO can't say I understand this training process; the PF loss keeps rising while the loss keeps falling even though I normalized the outputs;
    #  but at the end if falls a little bit again

    # Random seeds
    pyg.seed_everything(123)

    # Hyperparameters
    num_epochs = 1000
    batch_size = 512
    hidden_channels = 100
    num_layers = 3
    learning_rate = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ppcs and ppc -> Data

    solved_ppc_list = autodetect_load_ppc("generated_ppcs", shuffle=True, max_samples=5000)

    pf_data_list = []
    for solved_ppc in tqdm(solved_ppc_list, ascii=True, desc="Converting ppcs to data"):
        pf_data_list.append(power_flow_data(solved_ppc))

    for data in pf_data_list:
        data.x[~data.feature_mask] = 0.0  # delete the target info from the input features

    data_train, data_val = train_test_split(pf_data_list, test_size=0.33, random_state=42)

    # Torch dataloaders

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    node_features_stacked = torch.vstack([data.x for data in data_train])
    train_targets = torch.vstack([data.target_vector for data in data_train])

    # the output variables are of varying orders of magnitude so not normalizing them will the MSE loss
    # favoring larger variables and the weights in the final layer being of varying orders of magnitude themselves.
    # when calculating the power flow loss, an inverse transform needs to be applied to the predictions
    output_scaler = StandardScaler(train_targets)
    output_scaler.to(device)

    # Model

    standard_scaler = StandardScaler(node_features_stacked)
    model = GNN(in_channels=node_features_stacked.shape[1],
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=node_features_stacked.shape[1],
                standard_scaler=standard_scaler)
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
            batch = batch.to(device)

            optimizer.zero_grad()

            predictions = model(batch)
            loss = criterion(predictions, output_scaler(batch.target_vector))
            loss.backward()

            train_logs.append(supervised_pf_logs(loss, output_scaler.inverse(predictions), batch))

            optimizer.step()

        # Validation
        with torch.no_grad():

            model.eval()
            for batch in val_loader:
                batch = batch.to(device)

                predictions = model(batch)
                loss = criterion(predictions, output_scaler(batch.target_vector))

                val_logs.append(supervised_pf_logs(loss, output_scaler.inverse(predictions), batch))

        # Logging
        loss_str, r2score_str, mean_rel_P_error_str, median_rel_P_error_str = logs2str(train_logs, val_logs)

        # the PF errors reach a minimum but then start climbing again while the loss keeps declining because the loss focuses on minimizing some larger scale
        # components of itself while neglecting the smaller ones which has a noticeable effect on the PF.
        progress_bar.set_description(
            f"Training | Validation: {loss_str}; {r2score_str}; relative active power error: {mean_rel_P_error_str}, {median_rel_P_error_str}"
        )


if __name__ == '__main__':
    main()
