import pdb

import einops as E
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from .building_block import LinearBlock


# class LargeFeatureExtractor(torch.nn.Sequential):
#     def __init__(self, input_dim, output_dim):
#         super(LargeFeatureExtractor, self).__init__()
#         self.add_module('linear1', torch.nn.Linear(input_dim, 32))
#         self.add_module('relu1', torch.nn.ReLU())
#         self.add_module('linear2', torch.nn.Linear(32, 64))
#         self.add_module('relu2', torch.nn.ReLU())
#         self.add_module('linear3', torch.nn.Linear(64, 64))
#         self.add_module('relu3', torch.nn.ReLU())
#         self.add_module('linear4', torch.nn.Linear(64, 64))
#         self.add_module('relu4', torch.nn.ReLU())
#         self.add_module('linear5', torch.nn.Linear(64, output_dim))
#         self.add_module('final_act', torch.nn.Tanh())


class LargeFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(LargeFeatureExtractor, self).__init__()
        self.net = LinearBlock(
            in_features=config['rule_dim'],
            feature_list=config['gp_feat_list'] + [config['rbf_size']],
            act_inner=config['gp_feat_act'],
            act_out=config['gp_act_out']
        )

    def forward(self, x):
        return self.net(x)


class DGP(nn.Module):
    def __init__(self, config, eps_range=(1e-4, 1e-5)):
        super(DGP, self).__init__()
        self.z_dim = config['latent_view_size']
        self.rule_dim = config['rule_dim']
        self.rbf_coef_sigma = 1
        self.rbf_coef_l = 1
        self.rbf_size = config['rbf_size']
        self.extractors = nn.ModuleList()
        for i in range(self.z_dim):
            tmp_net = LargeFeatureExtractor(config)
            self.extractors.append(tmp_net)

    def rbf(self, x):
        batch_size, z_dim, total_points, d = x.shape
        diff = x.view(batch_size, z_dim, total_points, 1, d) - x.view(batch_size, z_dim, 1, total_points, d)
        product = (diff ** 2).sum(-1)
        return (self.rbf_coef_sigma ** 2) * torch.exp(- 0.5 * product / (self.rbf_coef_l ** 2))

    def add_noise(self, x):
        n = torch.FloatTensor(x.size()).uniform_(-1, 1).to(x.device)
        scale = x.detach() / 20
        return x + scale * n

    def forward(self, point_y, point_x, target_x, test_data=False):
        batch_size, observed_points, z_dim = point_y.shape
        lambda_latent = torch.cat((point_x, target_x), dim=1)
        totol_points, rule_dim = lambda_latent.shape[1], lambda_latent.shape[-1]
        features = []
        for i in range(z_dim):
            features.append(
                self.extractors[i](lambda_latent[:, :, i, :]).view(batch_size, 1, totol_points, self.rbf_size)
            )

        features = torch.cat(features, dim=1)
        features = self.add_noise(features)
        kernels = self.rbf(features)
        if test_data:
            cn = kernels[:, :, :observed_points, :observed_points]
            c = kernels[:, :, observed_points:, observed_points:]
            kt = kernels[:, :, observed_points:, :observed_points]
            k = kernels[:, :, :observed_points, observed_points:]
            point_y_reshape = E.rearrange(point_y, "b t d -> b d t 1")
            mu = torch.matmul(kt, torch.solve(point_y_reshape, cn)[0]).view(batch_size, z_dim, -1)
            Sigma = c - torch.matmul(kt, torch.solve(k, cn)[0])
            Sigma = torch.where(Sigma < 1e-5, torch.ones(Sigma.size()).to(mu) * 1e-5, Sigma)
            target_y = []
            for i in range(z_dim):
                target_y.append(MultivariateNormal(mu[:, i, :], Sigma[:, i, :, :]).rsample().unsqueeze(1))
            target_y = torch.cat(target_y, 1)

            target_y = E.rearrange(target_y, "b d t -> b t d")
        else:
            c_t = kernels[:, :, :observed_points, :observed_points]
            k_top_list = []
            k_list = []
            c_q_list = []
            for i in range(observed_points, totol_points):
                k_top_list.append(kernels[:, :, i:i + 1, :observed_points])
                k_list.append(kernels[:, :, :observed_points, i:i + 1])
                c_q_list.append(kernels[:, :, i:i + 1, i:i + 1])
            mu_list = []
            var_list = []
            point_y_reshape = E.rearrange(point_y, "b t d -> b d t 1")
            for i in range(totol_points - observed_points):
                mu_list.append(
                    torch.matmul(k_top_list[i], torch.solve(point_y_reshape, c_t)[0]).view(batch_size, z_dim))
                var_list.append(
                    c_q_list[i].view(batch_size, z_dim) - torch.matmul(k_top_list[i],
                                                                       torch.solve(k_list[i], c_t)[0]).view(
                        batch_size, z_dim))
            mu = torch.stack(mu_list, dim=1)
            sigma = torch.sqrt(torch.stack(var_list, dim=1))
            sigma = torch.where(sigma < 1e-5, torch.ones(sigma.size(), device=mu.device) * 1e-5, sigma)
            view_dist = Normal(mu, sigma)
            target_y = view_dist.rsample()

        kernels_diag = (torch.eye(kernels.shape[-1]).float()
                        .to(kernels.device)[None, None].
                        expand(kernels.shape[0], kernels.shape[1], -1, -1) * kernels).sum(-1)
        # c_q = kernels[:, :, observed_points:, observed_points:]
        # k_top = kernels[:, :, observed_points:, :observed_points]
        # k = kernels[:, :, :observed_points, observed_points:]
        # I = torch.eye(observed_points, device=c_t.device)[None, None].expand(batch_size, z_dim, -1, -1)
        # c_t_inverse = torch.solve(I, c_t)[0]
        # point_y_reshape = E.rearrange(point_y, "b t d -> b d t")
        # matrix = torch.einsum("bdqt,bdtn->bdqn", k_top, c_t_inverse)
        # mu = torch.einsum("bdmn,bdn->bdm", matrix, point_y_reshape)
        # mu = E.rearrange(mu, "b d t -> (b d) t")
        # matrix = torch.einsum("bdqt,bdtn->bdqn", k_top, c_t_inverse)
        # Sigma = c_q - torch.einsum("bdqt,bdtn", matrix, k)
        # Sigma = E.rearrange(Sigma, "b d m n -> (b d) m n")
        # dist = MultivariateNormal(mu, Sigma)
        # target_y = dist.rsample()
        # target_y = E.rearrange(target_y, "(b d) t -> b t d", b=batch_size, d=z_dim)
        if test_data:
            output = {'view_latent': target_y, 'diag': kernels_diag, 'kernel': kernels}
        else:
            output = {'q_view': view_dist, 'view_latent': target_y, 'view_mu': mu, 'view_logvar': 2 * torch.log(sigma),
                      'diag': kernels_diag, 'kernel': kernels}
        return output

    def predict(self, point_y, point_x, target_x):
        batch_size, observed_points, z_dim = point_y.shape
        lambda_latent = torch.cat((point_x, target_x), dim=1)
        totol_points, rule_dim = lambda_latent.shape[1], lambda_latent.shape[-1]
        features = []
        for i in range(z_dim):
            features.append(
                self.extractors[i](lambda_latent[:, :, i, :]).view(batch_size, 1, totol_points, self.rbf_size)
            )
        features = torch.cat(features, dim=1)
        features = self.add_noise(features)
        kernels = self.rbf(features)
        cn = kernels[:, :, :observed_points, :observed_points]
        c = kernels[:, :, observed_points:, observed_points:]
        kt = kernels[:, :, observed_points:, :observed_points]
        k = kernels[:, :, :observed_points, observed_points:]
        point_y_reshape = E.rearrange(point_y, "b t d -> b d t 1")
        mu = torch.matmul(kt, torch.solve(point_y_reshape, cn)[0]).view(batch_size, z_dim, -1)
        Sigma = c - torch.matmul(kt, torch.solve(k, cn)[0])
        Sigma = torch.where(Sigma < 1e-5, torch.ones(Sigma.size()).to(mu) * 1e-5, Sigma)
        target_y = []
        for i in range(z_dim):
            target_y.append(MultivariateNormal(mu[:, i, :], Sigma[:, i, :, :]).rsample().unsqueeze(1))
        target_y = torch.cat(target_y, 1)

        target_y = E.rearrange(target_y, "b d t -> b t d")
        return target_y
