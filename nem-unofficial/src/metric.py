import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


def convert_data(phase_param, data):
    num_steps = phase_param['num_steps']
    for key, val in data.items():
        if val.shape[1] == 1:
            data[key] = val.squeeze(1)
        else:
            data[key] = val[:, num_steps]
    return data


def compute_ami(config, data, results_all):
    seg_overlap = config['seg_overlap']
    segments = data['segment']
    overlaps = data['overlap']
    segments_valid = overlaps >= 1 if seg_overlap else overlaps == 1
    mask_all = results_all['mask']
    outputs = {key: [] for key in ['ami_obj']}
    for mask in mask_all:
        segment_o = np.argmax(mask, axis=1).squeeze(-1)
        sub_outputs = {key: [] for key in outputs}
        for seg_true, seg_valid, seg_o in zip(segments, segments_valid, segment_o):
            seg_true_sel = seg_true[seg_valid]
            seg_o_sel = seg_o[seg_valid]
            sub_outputs['ami_obj'].append(adjusted_mutual_info_score(seg_true_sel, seg_o_sel, average_method='max'))
        for key, val in sub_outputs.items():
            outputs[key].append(val)
    outputs = {key: np.array(val) for key, val in outputs.items()}
    return outputs


def compute_metrics(config, phase_param, data, results_all):
    data = convert_data(phase_param, data)
    metrics = compute_ami(config, data, results_all)
    return metrics
