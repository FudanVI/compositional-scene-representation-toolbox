import torch


def select_by_index(x, index):
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    x = torch.gather(x, index_ndim - 1, index)
    return x


def compute_variable_full(x_view, x_attr):
    x_view_expand = x_view[:, :, None].expand(-1, -1, x_attr.shape[1], -1)
    x_attr_expand = x_attr[:, None].expand(-1, x_view.shape[1], -1, -1)
    x = torch.cat([x_view_expand, x_attr_expand], dim=-1)
    return x


def reparameterize_normal(mu, logvar):
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    return mu + std * noise

def compute_kld_normal(mu, logvar, prior_mu, prior_logvar):
    prior_invvar = torch.exp(-prior_logvar)
    kld = 0.5 * (prior_logvar - logvar + prior_invvar * ((mu - prior_mu).square() + logvar.exp()) - 1)
    return kld.sum(-1)