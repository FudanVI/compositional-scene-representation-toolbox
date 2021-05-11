import numpy as np
from sklearn.metrics import adjusted_rand_score


def compute_ari(config, data, results_all):
    seg_overlap = config['seg_overlap']
    segments = data['segment']
    overlaps = data['overlap']
    segments_valid = overlaps >= 1 if seg_overlap else overlaps == 1
    mask_all = results_all['mask']
    outputs = {key: [] for key in ['ari_all']}
    for mask in mask_all:
        segment_a = np.argmax(mask, axis=1).squeeze(-1)
        sub_outputs = {key: [] for key in outputs}
        for seg_true, seg_valid, seg_a in zip(segments, segments_valid, segment_a):
            seg_true_sel = seg_true[seg_valid]
            seg_a_sel = seg_a[seg_valid]
            sub_outputs['ari_all'].append(adjusted_rand_score(seg_true_sel, seg_a_sel))
        for key, val in sub_outputs.items():
            outputs[key].append(val)
    outputs = {key: np.array(val) for key, val in outputs.items()}
    return outputs


def compute_metrics(config, data, results_all):
    metrics = compute_ari(config, data, results_all)
    return metrics
