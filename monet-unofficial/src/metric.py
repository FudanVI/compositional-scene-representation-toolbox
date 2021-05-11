import numpy as np
from sklearn.metrics import adjusted_rand_score


def compute_ari(config, data, results_all, key_mask):
    seg_overlap = config['seg_overlap']
    segments = data['segment']
    overlaps = data['overlap']
    segments_valid = overlaps >= 1 if seg_overlap else overlaps == 1
    mask_all = results_all[key_mask]
    outputs = {key: [] for key in ['ari_all', 'ari_obj']}
    for mask in mask_all:
        segment_a = np.argmax(mask, axis=1).squeeze(-1)
        segment_o = np.argmax(mask[:, 1:], axis=1).squeeze(-1)
        sub_outputs = {key: [] for key in outputs}
        for seg_true, seg_valid, seg_a, seg_o in zip(segments, segments_valid, segment_a, segment_o):
            seg_true_sel = seg_true[seg_valid]
            seg_a_sel = seg_a[seg_valid]
            seg_o_sel = seg_o[seg_valid]
            sub_outputs['ari_all'].append(adjusted_rand_score(seg_true_sel, seg_a_sel))
            sub_outputs['ari_obj'].append(adjusted_rand_score(seg_true_sel, seg_o_sel))
        for key, val in sub_outputs.items():
            outputs[key].append(val)
    outputs = {key: np.array(val) for key, val in outputs.items()}
    return outputs


def compute_metrics(config, data, results_all):
    metrics = compute_ari(config, data, results_all, 'mask')
    metrics_att = compute_ari(config, data, results_all, 'mask_att')
    metrics.update({key + '_att': val for key, val in metrics_att.items()})
    return metrics
