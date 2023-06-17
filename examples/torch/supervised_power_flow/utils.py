# from typing import Dict, List
#
# import numpy as np
# import torch
# from torch import Tensor
# from torch_geometric.data import Data
# from torcheval.metrics.functional import r2_score
#
# from mlpf.data.data.torch.power_flow import get_relative_power_flow_errors
#
#
# def list_of_dicts2dict_of_lists(list_of_dicts: List[Dict]) -> Dict:
#     """
#     Turn a list of dictionaries to a dictionary of lists.
#     :param list_of_dicts:
#     :return:
#     """
#     return {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}
#
#
# def supervised_pf_logs(loss: float, predictions: Tensor, batch: Data) -> Dict:
#     """
#     Log metrics for supervised power flow learning.
#
#     :param loss: Mean squared error loss value.
#     :param predictions: Prediction tensor.
#     :param batch: PyG data batch.
#     :return: Logs dictionary.
#     """
#     predictions = predictions.detach().cpu()
#     batch = batch.detach().cpu()
#     relative_active_power_errors, relative_reactive_power_errors = get_relative_power_flow_errors(predictions, batch)
#     r2score = r2_score(predictions, batch.target_vector, multioutput="raw_values").detach()
#     r2score[torch.isinf(r2score)] = 0.0
#
#     return {
#         "loss": loss.item(),
#         "r2_score": torch.mean(r2score).item(),
#         "rel P error mean": torch.mean(relative_active_power_errors).item(),
#         "rel P error median": torch.median(relative_active_power_errors).item()
#     }
#
#
# def logs2str(train_logs: List[Dict], val_logs: List[Dict]):
#     """
#     Return the nicely formatted metric stings for command line printing.
#
#     :param train_logs:
#     :param val_logs:
#     :return:
#     """
#     train_log = list_of_dicts2dict_of_lists(train_logs)
#     val_log = list_of_dicts2dict_of_lists(val_logs)
#
#     loss_str = f"loss: ({np.mean(train_log['loss']):2.4f} | {np.mean(val_log['loss']):2.4f})"
#     r2score_str = f"r2score: ({np.mean(train_log['r2_score']):2.4f} | {np.mean(val_log['r2_score']):2.4f})"
#     mean_rel_P_error_str = f"mean: ({np.mean(train_log['rel P error mean']):2.4f} | {np.mean(val_log['rel P error mean']):2.4f})"
#     median_rel_P_error_str = f"median: ({np.mean(train_log['rel P error median']):2.4f} | {np.mean(val_log['rel P error median']):2.4f})"
#
#     return loss_str, r2score_str, mean_rel_P_error_str, median_rel_P_error_str
