import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


def compute_mse(data, results_all):
    images = data['image']
    recon_all = results_all['recon']
    mse_all = []
    for recon in recon_all:
        mse = np.square(recon - images).reshape(images.shape[0], -1).mean(-1)
        mse_all.append(mse)
    return np.array(mse_all)


def compute_ll(config, data, results_all, eps=1e-10):
    normal_scale = config['normal_scale']
    normal_invvar = 1 / pow(normal_scale, 2)
    normal_const = np.log(2 * np.pi / normal_invvar)
    images = data['image']
    mask_all = results_all['mask']
    apc_all = results_all['apc']
    ll_all = []
    for mask, apc in zip(mask_all, apc_all):
        log_mask = np.log(mask + eps)
        raw_pixel_ll = -0.5 * (normal_const + normal_invvar * np.square(apc - images[:, None])).sum(-1, keepdims=True)
        log_prob = torch.from_numpy(log_mask + raw_pixel_ll)
        pixel_ll = torch.logsumexp(log_prob, dim=1)
        ll = pixel_ll.sum([*range(1, pixel_ll.ndim)]).numpy()
        ll_all.append(ll)
    return np.array(ll_all)


def compute_oca(data, results_all):
    segments = data['segment']
    seg_back = segments.max()
    counts = np.array([(np.unique(val) != seg_back).sum() for val in segments])
    pres_all = results_all['pres']
    oca_all = []
    for pres in pres_all:
        oca = ((pres[:, :-1] >= 0.5).sum(-1) == counts).astype(np.float)
        oca_all.append(oca)
    return np.array(oca_all)


def compute_ari_ami(config, data, results_all):
    seg_overlap = config['seg_overlap']
    segments = data['segment']
    overlaps = data['overlap']
    segments_valid = overlaps >= 1 if seg_overlap else overlaps == 1
    mask_all = results_all['mask']
    outputs = {key: [] for key in ['ari_all', 'ari_obj', 'ami_all', 'ami_obj']}
    for mask in mask_all:
        segment_a = np.argmax(mask, axis=1).squeeze(-1)
        segment_o = np.argmax(mask[:, :-1], axis=1).squeeze(-1)
        sub_outputs = {key: [] for key in outputs}
        for seg_true, seg_valid, seg_a, seg_o in zip(segments, segments_valid, segment_a, segment_o):
            seg_true_sel = seg_true[seg_valid]
            seg_a_sel = seg_a[seg_valid]
            seg_o_sel = seg_o[seg_valid]
            sub_outputs['ari_all'].append(adjusted_rand_score(seg_true_sel, seg_a_sel))
            sub_outputs['ari_obj'].append(adjusted_rand_score(seg_true_sel, seg_o_sel))
            sub_outputs['ami_all'].append(adjusted_mutual_info_score(seg_true_sel, seg_a_sel, average_method='max'))
            sub_outputs['ami_obj'].append(adjusted_mutual_info_score(seg_true_sel, seg_o_sel, average_method='max'))
        for key, val in sub_outputs.items():
            outputs[key].append(val)
    outputs = {key: np.array(val) for key, val in outputs.items()}
    return outputs


def compute_order(cost_all):
    order_all = []
    for cost_list in cost_all:
        order_list = []
        for cost in cost_list:
            _, cols = linear_sum_assignment(cost)
            order_list.append(cols)
        order_all.append(order_list)
    return np.array(order_all)


def select_by_order(val_all, order_all):
    val_all_sel = []
    for val_list, order_list in zip(val_all, order_all):
        val_sel = np.array([val[order] for val, order in zip(val_list, order_list)])
        val_sel = np.concatenate([val_sel, val_list[:, -1:]], axis=1)
        val_all_sel.append(val_sel)
    return np.array(val_all_sel)


def compute_iou_f1_full(seg_true, seg_all, eps=1e-6):
    iou_all, f1_all, weights_all = [], [], []
    for seg_pred in seg_all:
        seg_true = seg_true.reshape(*seg_true.shape[:2], -1)
        seg_pred = seg_pred.reshape(*seg_pred.shape[:2], -1)
        pres = (seg_true.max(-1) != 0).astype(np.float32)
        area_i = np.minimum(seg_true, seg_pred).sum(-1)
        area_u = np.maximum(seg_true, seg_pred).sum(-1)
        iou = area_i / (area_u + eps)
        f1 = 2 * area_i / (area_i + area_u + eps)
        iou_all.append((iou * pres).sum(-1))
        f1_all.append((f1 * pres).sum(-1))
        weights_all.append(pres.sum(-1))
    iou_all = np.array(iou_all)
    f1_all = np.array(f1_all)
    weights_all = np.array(weights_all)
    return (iou_all, weights_all), (f1_all, weights_all)


def compute_iou_f1_part(config, data, results_all):
    seg_overlap = config['seg_overlap']
    overlaps = data['overlap'][:, None, ..., None]
    mask_valid = np.ones(overlaps.shape) if seg_overlap else (overlaps <= 1).astype(np.float)
    mask_all = results_all['mask']
    mask_all *= mask_valid[None]
    if 'layers' in data:
        shp_true = data['layers'][..., -1:]
        part_cumprod = np.concatenate([
            np.ones((shp_true.shape[0], 1, *shp_true.shape[2:])),
            np.cumprod(1 - shp_true[:, :-1], 1),
        ], axis=1)
        mask_true = shp_true * part_cumprod
    else:
        segments = torch.from_numpy(data['segment'])[:, None, ..., None]
        scatter_shape = [segments.shape[0], segments.max() + 1, *segments.shape[2:]]
        mask_true = torch.zeros(scatter_shape).scatter_(1, segments, 1).numpy().astype(np.float)
    mask_true *= mask_valid
    order_cost = -(mask_true[None, :, :-1, None] * mask_all[:, :, None, :-1])
    order_cost = order_cost.reshape(*order_cost.shape[:-3], -1).sum(-1)
    order_all = compute_order(order_cost)
    mask_all_sel = select_by_order(mask_all, order_all)
    iou_all, f1_all = compute_iou_f1_full(mask_true, mask_all_sel)
    return iou_all, f1_all, order_all


def compute_ooa(data, order_all):
    layers = data['layers']
    objects_apc, objects_shp = layers[:, :-1, ..., :-1], layers[:, :-1, ..., -1:]
    weights = np.zeros((objects_shp.shape[0], objects_shp.shape[1], objects_shp.shape[1]))
    for i in range(objects_shp.shape[1] - 1):
        for j in range(i + 1, objects_shp.shape[1]):
            sq_diffs = np.square(objects_apc[:, i] - objects_apc[:, j]).sum(-1, keepdims=True)
            sq_diffs *= objects_shp[:, i] * objects_shp[:, j]
            weights[:, i, j] = sq_diffs.reshape(sq_diffs.shape[0], -1).sum(-1)
    ooa_all, weights_all = [], []
    for order in order_all:
        binary_mat = np.zeros(weights.shape)
        for i in range(order.shape[1] - 1):
            for j in range(i + 1, order.shape[1]):
                binary_mat[:, i, j] = order[:, i] < order[:, j]
        ooa_all.append((binary_mat * weights).reshape(weights.shape[0], -1).sum(-1))
        weights_all.append(weights.reshape(weights.shape[0], -1).sum(-1))
    return np.array(ooa_all), np.array(weights_all)


def compute_layer_mse(layers, apc_all, shp_all):
    target_apc, target_shp = layers[..., :-1], layers[..., -1:]
    layer_mse_all, weights_all = [], []
    for apc, shp in zip(apc_all, shp_all):
        noise = np.random.uniform(0, 1, size=apc.shape)
        target_recon = target_apc * target_shp + noise * (1 - target_shp)
        recon = apc * shp + noise * (1 - shp)
        sq_diffs = np.square(recon - target_recon).mean(-1, keepdims=True)
        mask_valid = np.maximum(target_shp, shp)
        layer_mse_all.append((sq_diffs * mask_valid).reshape(mask_valid.shape[0], -1).mean(-1))
        weights_all.append(mask_valid.reshape(mask_valid.shape[0], -1).mean(-1))
    return np.array(layer_mse_all), np.array(weights_all)


def compute_metrics(config, data, results_all):
    metrics = {
        'mse': compute_mse(data, results_all),
        'll': compute_ll(config, data, results_all),
        'oca': compute_oca(data, results_all),
    }
    metrics.update(compute_ari_ami(config, data, results_all))
    metrics['iou_part'], metrics['f1_part'], order_all = compute_iou_f1_part(config, data, results_all)
    metrics['ooa'] = compute_ooa(data, order_all)
    if 'layers' in data:
        layers = data['layers']
        shp_true = layers[..., -1:]
        apc_all_sel = select_by_order(results_all['apc'], order_all)
        shp_all_sel = select_by_order(results_all['shp'], order_all)
        metrics['iou_full'], metrics['f1_full'] = compute_iou_f1_full(shp_true, shp_all_sel)
        metrics['layer_mse'] = compute_layer_mse(layers, apc_all_sel, shp_all_sel)
    return metrics
