import os
import pdb
import pickle

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def compute_order(data, results):
    segment = torch.from_numpy(data['segment'])[:, :, None, ..., None]
    scatter_shape = [*segment.shape[:2], segment.max() + 1, *segment.shape[3:]]
    obj_mask_true = torch.zeros(scatter_shape).scatter_(2, segment, 1).numpy().astype(np.float64)[:, :, :-1]
    obj_shp_true = data['masks'][:, :, :-1]
    binary_mat_true = np.zeros((*obj_shp_true.shape[:2], obj_shp_true.shape[2], obj_shp_true.shape[2]))
    for i in range(obj_shp_true.shape[2] - 1):
        for j in range(i + 1, obj_shp_true.shape[2]):
            region = np.minimum(obj_shp_true[:, :, i], obj_shp_true[:, :, j])
            area_i = (obj_mask_true[:, :, i] * region).reshape(*region.shape[:2], -1).sum(-1)
            area_j = (obj_mask_true[:, :, j] * region).reshape(*region.shape[:2], -1).sum(-1)
            binary_mat_true[:, :, i, j] = (area_i >= area_j) * 2 - 1
    obj_mask_all = results['mask'][:, :, :, :-1]
    order_cost_all = -(obj_mask_true[None, :, :, :, None] * obj_mask_all[:, :, :, None])
    order_cost_all = order_cost_all.reshape(*order_cost_all.shape[:-3], -1).sum(-1)
    order_cost_all = order_cost_all.sum(2)
    order_all = []
    for cost_list in order_cost_all:
        order_list = []
        for cost in cost_list:
            _, cols = linear_sum_assignment(cost)
            order_list.append(cols)
        order_all.append(order_list)
    order_all = np.array(order_all)
    return order_all, binary_mat_true


def compute_oca(data, results):
    return None


def compute_ooa(data, results, order_all, binary_mat_true):
    obj_shp_true = data['masks'][:, :, :-1]
    weights = np.zeros((*obj_shp_true.shape[:2], obj_shp_true.shape[2], obj_shp_true.shape[2]))
    for i in range(obj_shp_true.shape[2] - 1):
        for j in range(i + 1, obj_shp_true.shape[2]):
            region = np.minimum(obj_shp_true[:, :, i], obj_shp_true[:, :, j])
            weights[:, :, i, j] = region.reshape(*region.shape[:2], -1).sum(-1)
    sum_weights = weights.sum()
    log_ord_all = results['log_ord']
    ooa_all = []
    for order, log_ord in zip(order_all, log_ord_all):
        binary_mat_pred = np.zeros(weights.shape)
        for idx_data in range(weights.shape[0]):
            sub_order = order[idx_data]
            sub_log_ord = log_ord[idx_data]
            for idx_view in range(weights.shape[1]):
                sub_sub_log_ord = sub_log_ord[idx_view]
                for i in range(order.shape[1] - 1):
                    idx_i = sub_order[i]
                    for j in range(i + 1, order.shape[1]):
                        idx_j = sub_order[j]
                        binary_mat_pred[idx_data, idx_view, i, j] = (sub_sub_log_ord[idx_i] > sub_sub_log_ord[
                            idx_j]) * 2 - 1
        binary_mat = (binary_mat_true * binary_mat_pred) == 1
        ooa_all.append((binary_mat * weights).sum() / sum_weights)
    ooa_all = np.array(ooa_all)
    return ooa_all


def compute_ari_ami(data, results):
    segment_true = data['segment']
    overlap = data['overlap']
    segment_sel = overlap >= 1
    mask_all = results['mask']
    outputs = {key: [] for key in ['ari_all', 'ari_obj', 'ami_all', 'ami_obj']}
    for mask in mask_all:
        segment_a = np.argmax(mask, axis=2).squeeze(-1)
        segment_o = np.argmax(mask[:, :, :-1], axis=2).squeeze(-1)
        sub_outputs = {key: [] for key in outputs}
        for seg_true, seg_sel, seg_a, seg_o in zip(segment_true, segment_sel, segment_a, segment_o):
            seg_a_true_sel = seg_true.reshape(-1)
            seg_o_true_sel = seg_true[seg_sel]
            seg_a_sel = seg_a.reshape(-1)
            seg_o_sel = seg_o[seg_sel]
            sub_outputs['ari_all'].append(adjusted_rand_score(seg_a_true_sel, seg_a_sel))
            sub_outputs['ari_obj'].append(adjusted_rand_score(seg_o_true_sel, seg_o_sel))
            sub_outputs['ami_all'].append(
                adjusted_mutual_info_score(seg_a_true_sel, seg_a_sel, average_method='arithmetic'))
            sub_outputs['ami_obj'].append(
                adjusted_mutual_info_score(seg_o_true_sel, seg_o_sel, average_method='arithmetic'))
        for key, val in sub_outputs.items():
            outputs[key].append(val)
    outputs = {key: np.array(val).mean(-1) for key, val in outputs.items()}
    return outputs


def select_by_index(x, index_raw):
    x = torch.from_numpy(x)
    index = torch.from_numpy(index_raw)
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    if index_raw.ndim == 2:
        x_obj = torch.gather(x[:, :-1], index_ndim - 1, index)
        x = torch.cat([x_obj, x[:, -1:]], dim=1)
    elif index_raw.ndim == 3:
        x_obj = torch.gather(x[:, :, :-1], index_ndim - 1, index)
        x = torch.cat([x_obj, x[:, :, -1:]], dim=2)
    elif index_raw.ndim == 4:
        x_obj = torch.gather(x[:, :, :, :-1], index_ndim - 1, index)
        x = torch.cat([x_obj, x[:, :, :, -1:]], dim=3)
    else:
        raise AssertionError
    return x.numpy()


def compute_iou_f1(data, results, order_all, eps=1e-6):
    obj_shp_true = data['masks'][:, :, :-1]
    order_all = order_all[:, :, None].repeat(obj_shp_true.shape[1], axis=2)
    obj_shp_all = select_by_index(results['shp'], order_all)[:, :, :, :-1]
    seg_true = obj_shp_true.reshape(*obj_shp_true.shape[:3], -1)
    pres = (seg_true.max(-1).max(1) != 0).astype(np.float64)
    sum_pres = pres.sum()
    outputs = {key: [] for key in ['iou', 'f1']}
    for obj_shp in obj_shp_all:
        seg_pred = obj_shp.reshape(*obj_shp.shape[:3], -1)
        area_i = np.minimum(seg_true, seg_pred).sum(-1).sum(1)
        area_u = np.maximum(seg_true, seg_pred).sum(-1).sum(1)
        iou = area_i / np.clip(area_u, eps, None)
        f1 = 2 * area_i / np.clip(area_i + area_u, eps, None)
        outputs['iou'].append((iou * pres).sum() / sum_pres)
        outputs['f1'].append((f1 * pres).sum() / sum_pres)
    outputs = {key: np.array(val) for key, val in outputs.items()}
    return outputs

def select(x, index_raw):
    x = torch.from_numpy(x)
    index = torch.tensor(index_raw,dtype=torch.int64)
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    if index_raw.ndim == 2:
        x = torch.gather(x, index_ndim - 1, index)
    elif index_raw.ndim == 3:
        x = torch.gather(x, index_ndim - 1, index)
    elif index_raw.ndim == 4:
        x = torch.gather(x, index_ndim - 1, index)
    else:
        raise AssertionError
    return x.numpy()

def compute_mse(data, results):
    image = torch.from_numpy(data['image'][None])
    recon = torch.from_numpy(results['recon'])
    diff = (image - recon).square()
    mse = diff.mean([*range(1, diff.ndim)]).numpy()
    return mse




folder_out = 'metrics'
folder_data = '/home/ctl/conference/gcm/dataset'
mode_list = ['simple', 'complex']
data_list = ['clevr', 'shop']
data_views_list = [10]
name_list = ['{}_multi_{}_{}'.format(name, mode, data_views) for name in data_list for mode in mode_list for data_views
             in data_views_list]
checkpoint_list = ['../../outs/gp_two_stage_transformer/{}/random_length/two/predict'.format(name) for name
                   in name_list]
if not os.path.exists(folder_out):
    os.makedirs(folder_out)
phase_list = ['test']
for observed_views in [2, 3, 4, 5, 6, 7, 8, 9]:
    metrics = {}
    for name_data, checkpoint in zip(name_list, checkpoint_list):
        metrics[name_data] = {}
        for phase in phase_list:
            metrics[name_data][phase] = {}
            data_root = '_'.join(name_data.split('_')[:2])
            with h5py.File(os.path.join(folder_data, data_root, '{}.h5'.format(name_data)), 'r') as f:
                data = {key: f[phase][key][:, :10] for key in f[phase]}
                for key, val in data.items():
                    if key in ['segment', 'overlap']:
                        data[key] = val.astype(np.int64)
                    else:
                        data[key] = val.astype(np.float64) / 255
            for mode in [1, 2]:
                results = {}
                with h5py.File(os.path.join(checkpoint, '{}_m{}_statistics_o{}.h5'.format(phase, mode, observed_views)), 'r') as f:
                    for key in f['Q']:
                        if key in ['pres']:
                            results.update({key: f['T'][key][()] / 255})
                        if key in ['recon', 'mask', 'shp']:
                            results.update({key: f['Q'][key][()] / 255})
                        elif key in ['index', 'log_ord']:
                            results.update({key: f['Q'][key][()]})
                query_view = 1
                print(
                    'processing dataset: {}, phase {}, mode {}, observe {},query {} '.format(name_data, phase, mode,
                                                                                             observed_views,
                                                                                             query_view))
                sub_results = {}
                for key in results:
                    if key not in ['pres']:
                        sub_results.update({key: results[key][:, :, :query_view]})
                    else:
                        sub_results.update({key: results[key]})
                index = sub_results['index'][0]
                sub_data = {key: select(val, index) for key, val in data.items()}
                style = 'm{}_q{}'.format(mode, query_view)
                metrics[name_data][phase][style] = {}
                order_all, binary_mat_true = compute_order(sub_data, sub_results)
                metrics[name_data][phase][style]['mse'] = compute_mse(sub_data, sub_results)
                metrics[name_data][phase][style].update(compute_iou_f1(sub_data, sub_results, order_all))
                # metrics[name_data][phase][style]['oca'] = compute_oca(data, sub_results)
                metrics[name_data][phase][style]['ooa'] = compute_ooa(sub_data, sub_results, order_all, binary_mat_true)
                metrics[name_data][phase][style].update(compute_ari_ami(sub_data, sub_results))
    with open('statistics_metrics_{}.pkl'.format(observed_views), 'wb') as f:
        pickle.dump(metrics, f)
