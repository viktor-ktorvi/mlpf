import hydra
from matplotlib import pyplot as plt

from mlpf.data.analysis.utils import table_and_columns_from_config, create_subplots_grid
from mlpf.data.analysis.visualization.grid_pdf_estimates import visualize_grid_pdfs, visualize_grid_histograms
from mlpf.data.loading.load_data import autodetect_load_ppc
from mlpf.enumerations.bus_type import BusTypeIds


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    ppc_list = autodetect_load_ppc(cfg.data_path)

    bus_type = BusTypeIds(cfg.bus_type) if cfg.bus_type is not None else None

    table, columns = table_and_columns_from_config(cfg)

    num_columns = len(columns) if columns is not None else ppc_list[0][table.value].shape[1]
    fig, axes = create_subplots_grid(num_columns)

    fig.tight_layout()
    visualize_grid_pdfs(ppc_list, table, bus_type=bus_type, columns=columns, kernel=cfg.visualization.kernel, bandwidth_coeff=cfg.visualization.bandwidth_coeff, axes=axes)

    for ax in axes.flatten():
        ax.set_ylim(bottom=0)

    fig, axes = create_subplots_grid(num_columns)
    fig.tight_layout()

    visualize_grid_histograms(ppc_list, table, bus_type=bus_type, columns=columns, bins=cfg.visualization.bins, axes=axes)

    plt.show()


if __name__ == "__main__":
    main()
