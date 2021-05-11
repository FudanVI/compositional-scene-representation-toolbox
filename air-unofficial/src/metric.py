import numpy as np


def compute_ll(config, data, results_all):
    normal_scale = config['normal_scale']
    normal_invvar = 1 / pow(normal_scale, 2)
    normal_const = np.log(2 * np.pi / normal_invvar)
    images = data['image']
    recon_all = results_all['recon']
    ll_all = []
    for recon in recon_all:
        pixel_ll = -0.5 * (normal_const + normal_invvar * np.square(recon - images))
        ll = pixel_ll.reshape(pixel_ll.shape[0], -1).sum(1)
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


def compute_metrics(config, data, results_all):
    metrics = {
        'll': compute_ll(config, data, results_all),
        'oca': compute_oca(data, results_all),
    }
    return metrics
