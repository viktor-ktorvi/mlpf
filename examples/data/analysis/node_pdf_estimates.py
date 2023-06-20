import hydra
from matplotlib import pyplot as plt

from mlpf.data.analysis.utils import table_and_columns_from_config, create_subplots_grid
from mlpf.data.analysis.visualization.node_pdf_estimates import visualize_node_pdfs, visualize_node_histograms
from mlpf.data.loading.load_data import autodetect_load_ppc


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    ppc_list = autodetect_load_ppc(cfg.data_path)

    table, columns = table_and_columns_from_config(cfg)

    num_columns = len(columns) if columns is not None else ppc_list[0][table.value].shape[1]
    fig, axes = create_subplots_grid(num_columns)

    fig.tight_layout()
    visualize_node_pdfs(ppc_list, table, node_numbers=cfg.node_numbers, columns=columns, kernel=cfg.visualization.kernel, bandwidth_coeff=cfg.visualization.bandwidth_coeff,
                        axes=axes)

    for ax in axes.flatten():
        ax.set_ylim(bottom=0)

    fig, axes = create_subplots_grid(num_columns)
    fig.tight_layout()

    visualize_node_histograms(ppc_list, table, node_numbers=cfg.node_numbers, columns=columns, bins=cfg.visualization.bins, axes=axes)

    plt.show()


if __name__ == "__main__":
    main()
