import math
import sonnet as snt
import tensorflow.compat.v1 as tf
from network import AttentionNet, VAENet


def get_batch_size(x, name='batch_size'):
    with tf.name_scope(name):
        batch_size = x.shape[0].value
        if batch_size is None:
            batch_size = tf.shape(x)[0]
    return batch_size


class Model(snt.AbstractModule):

    def __init__(self, config, name='model'):
        super(Model, self).__init__(name=name)
        self._normal_invvar = {key: 1 / pow(val, 2) for key, val in config['normal_scale'].items()}
        self._normal_const = {key: math.log(2 * math.pi / val) for key, val in self._normal_invvar.items()}
        self._seg_overlap = config['seg_overlap']
        with self._enter_variable_scope(check_same_graph=False):
            self.net_att = AttentionNet(config)
            self.net_vae = VAENet(config)

    def _build(self, images, segments, overlaps, num_slots):
        batch_size = get_batch_size(images)
        variables = {}
        log_scope = tf.zeros([batch_size, *images.shape[1:-1], 1], name='init_log_scope')
        for slot in range(num_slots):
            with tf.name_scope('slot_{}'.format(slot)):
                if slot < num_slots - 1:
                    log_att, log1m_att = self.net_att(images, log_scope)
                    with tf.name_scope('log_att_mask'):
                        log_att_mask = log_scope + log_att
                    with tf.name_scope('log_scope'):
                        log_scope = log_scope + log1m_att
                else:
                    log_att_mask = log_scope
                apc, logits_mask, loc, raw_scale = self.net_vae(images, log_att_mask, batch_size)
                slot_variables = {'log_att_mask': log_att_mask, 'apc': apc, 'logits_mask': logits_mask,
                                  'loc': loc, 'raw_scale': raw_scale}
                for key, val in slot_variables.items():
                    if key in variables:
                        variables[key].append(val)
                    else:
                        variables[key] = [val]
        with tf.name_scope('stack'):
            variables = {key: tf.stack(val, axis=1) for key, val in variables.items()}
        with tf.name_scope('mask'):
            log_mask = tf.math.log_softmax(variables['logits_mask'], axis=1)
            mask = tf.math.softmax(variables['logits_mask'], axis=1)
            log_att_mask = variables['log_att_mask']
            att_mask = tf.math.exp(log_att_mask)
        losses = self._get_loss_values(images, variables, att_mask, log_mask)
        outputs = self._get_outputs(variables, mask, name='outputs')
        outputs_att = self._get_outputs(variables, att_mask, name='att_outputs')
        results = {'image': images, 'apc': variables['apc'], 'mask': mask, 'mask_att': att_mask}
        for key in ['pres', 'recon']:
            results[key] = outputs[key]
            results[key + '_att'] = outputs_att[key]
        metrics = self._get_metrics(
            images, segments, overlaps, results, outputs, outputs_att, log_mask, log_att_mask, batch_size)
        with tf.name_scope('loss_compare'):
            losses['compare'] = -metrics['ll'] + losses['kld']
        return results, metrics, losses

    def _get_log_likelihood(self, images, apc, log_mask, name='log_likelihood'):
        with tf.name_scope(name):
            sq_diff = tf.math.squared_difference(apc, tf.expand_dims(images, axis=1))
            sq_diff_bck, sq_diff_obj = tf.split(sq_diff, [1, sq_diff.shape.as_list()[1] - 1], axis=1)
            raw_pixel_ll_bck = -0.5 * (self._normal_const['bck'] + self._normal_invvar['bck'] * sq_diff_bck)
            raw_pixel_ll_obj = -0.5 * (self._normal_const['obj'] + self._normal_invvar['obj'] * sq_diff_obj)
            raw_pixel_ll = tf.concat([raw_pixel_ll_bck, raw_pixel_ll_obj], axis=1)
            raw_pixel_ll = tf.math.reduce_sum(raw_pixel_ll, axis=-1, keepdims=True)
            pixel_ll = tf.math.reduce_logsumexp(log_mask + raw_pixel_ll, axis=1)
            ll = tf.math.reduce_sum(pixel_ll, axis=[*range(1, pixel_ll.shape.rank)])
        return ll

    def _get_loss_values(self, images, variables, att_mask, log_mask, eps=1e-10, name='loss_values'):
        with tf.name_scope(name):
            apc = variables['apc']
            log_att_mask = variables['log_att_mask']
            loc = variables['loc']
            raw_scale = variables['raw_scale']
            with tf.name_scope('loss_nll'):
                ll = self._get_log_likelihood(images, apc, log_att_mask)
                loss_nll = -ll
            with tf.name_scope('loss_kld'):
                scale = tf.math.softplus(raw_scale)
                var = tf.math.square(scale)
                loss_kld = 0.5 * (tf.math.square(loc) + var - tf.math.log(var + eps) - 1)
                loss_kld = tf.math.reduce_sum(loss_kld, axis=[*range(1, loss_kld.shape.rank)])
            with tf.name_scope('loss_mask'):
                loss_mask = att_mask * (log_att_mask - log_mask)
                loss_mask = tf.math.reduce_sum(loss_mask, axis=[*range(1, loss_mask.shape.rank)])
        return {'nll': loss_nll, 'kld': loss_kld, 'mask': loss_mask}

    @staticmethod
    def _get_outputs(variables, mask, name='outputs'):
        with tf.name_scope(name):
            recon = tf.math.reduce_sum(mask * variables['apc'], axis=1)
            segment_all = tf.math.argmax(mask, axis=1)
            segment_obj = tf.math.argmax(mask[:, 1:], axis=1)
            mask_oh_all = tf.one_hot(segment_all, mask.shape[1], axis=1)
            mask_oh_obj = tf.one_hot(segment_obj, mask.shape[1], axis=1)
            pres = tf.math.reduce_max(mask_oh_all, axis=[-3, -2, -1])
            outputs = {'recon': recon, 'segment_all': segment_all, 'segment_obj': segment_obj,
                       'mask_oh_all': mask_oh_all, 'mask_oh_obj': mask_oh_obj, 'pres': pres}
        return outputs

    @staticmethod
    def _compute_ari(mask_true, mask_pred, batch_size, name='compute_ari'):
        def comb2(x):
            x = x * (x - 1)
            if x.shape.rank > 1:
                x = tf.math.reduce_sum(x, axis=[*range(1, x.shape.rank)])
            return x
        with tf.name_scope(name):
            num_pixels = tf.math.reduce_sum(mask_true, axis=[*range(1, mask_true.shape.rank)])
            mask_true = tf.reshape(
                mask_true, [batch_size, mask_true.shape[1], 1, mask_true.shape[2] * mask_true.shape[3]])
            mask_pred = tf.reshape(
                mask_pred, [batch_size, 1, mask_pred.shape[1], mask_pred.shape[2] * mask_pred.shape[3]])
            mat = tf.math.reduce_sum(mask_true * mask_pred, axis=-1)
            sum_row = tf.math.reduce_sum(mat, axis=1)
            sum_col = tf.math.reduce_sum(mat, axis=2)
            comb_mat = comb2(mat)
            comb_row = comb2(sum_row)
            comb_col = comb2(sum_col)
            comb_num = comb2(num_pixels)
            comb_prod = (comb_row * comb_col) / comb_num
            comb_mean = 0.5 * (comb_row + comb_col)
            diff = comb_mean - comb_prod
            score = (comb_mat - comb_prod) / diff
            invalid = tf.math.logical_or(tf.math.equal(comb_num, 0), tf.math.equal(diff, 0))
            score = tf.where(invalid, tf.ones_like(score), score)
        return score

    def _get_metrics(self, images, segments, overlaps, results, outputs, outputs_att, log_mask, log_att_mask,
                     batch_size, name='metrics'):
        with tf.name_scope(name):
            segments_obj = segments[:, :-1]
            segments_obj_sel = segments_obj if self._seg_overlap else segments_obj * (1 - overlaps)
            with tf.name_scope('ari'):
                ari_all = self._compute_ari(segments_obj_sel, outputs['mask_oh_all'], batch_size)
                ari_obj = self._compute_ari(segments_obj_sel, outputs['mask_oh_obj'], batch_size)
                ari_all_att = self._compute_ari(segments_obj_sel, outputs_att['mask_oh_all'], batch_size)
                ari_obj_att = self._compute_ari(segments_obj_sel, outputs_att['mask_oh_obj'], batch_size)
            with tf.name_scope('mse'):
                pixel_mse = tf.math.squared_difference(results['recon'], images)
                pixel_mse_att = tf.math.squared_difference(results['recon_att'], images)
                mse = tf.math.reduce_mean(pixel_mse, axis=[*range(1, pixel_mse.shape.rank)])
                mse_att = tf.math.reduce_mean(pixel_mse_att, axis=[*range(1, pixel_mse_att.shape.rank)])
            with tf.name_scope('ll'):
                ll = self._get_log_likelihood(images, results['apc'], log_mask)
                ll_att = self._get_log_likelihood(images, results['apc'], log_att_mask)
            with tf.name_scope('count_acc'):
                pres_true = tf.math.reduce_max(segments_obj, axis=[-3, -2, -1])
                count_true = tf.math.reduce_sum(pres_true, axis=1)
                count_pred = tf.math.reduce_sum(results['pres'][:, 1:], axis=1)
                count_pred_att = tf.math.reduce_sum(results['pres_att'][:, 1:], axis=1)
                count_acc = tf.cast(tf.math.equal(count_true, count_pred), dtype=tf.float32)
                count_acc_att = tf.cast(tf.math.equal(count_true, count_pred_att), dtype=tf.float32)
            metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'll': ll, 'count': count_acc,
                       'ari_all_att': ari_all_att, 'ari_obj_att': ari_obj_att, 'mse_att': mse_att, 'll_att': ll_att,
                       'count_att': count_acc_att}
        return metrics


def get_model(config):
    net = Model(config)
    return net
