import warnings

import hydra
import pandapower as pp
import pandapower.networks as pn

from mlpf.data.generate.generate_uniform_data import generate_uniform_ppcs
from mlpf.data.utils.saving import pickle_all


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    """
    Generate a dataset and save it. Use the hydra config default or overwrite the command line args.

    :param cfg: hydra config
    :return:
    """
    # net = pn.create_kerber_dorfnetz()  # TODO think of an elegant way to choose grids from the command line
    net = pn.case118()
    pp.runpp(net)
    base_ppc = net._ppc

    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

    uniform_ppc_list = generate_uniform_ppcs(
        base_ppc,
        how_many=cfg.how_many,
        low=cfg.low,
        high=cfg.high
    )

    pickle_all(uniform_ppc_list, save_path=cfg.save_path, extension=cfg.extension, delete_all_from_save_path=True)


if __name__ == "__main__":
    main()
