import math
import sonnet as snt
import tensorflow.compat.v1 as tf
from network import Initializer, Updater, Decoder


def get_batch_size(x, name='batch_size'):
    with tf.name_scope(name):
        batch_size = x.shape[0].value
        if batch_size is None:
            batch_size = tf.shape(x)[0]
    return batch_size


class Model(snt.AbstractModule):

    def __init__(self, config, name='model'):
        super(Model, self).__init__(name=name)
        self._normal_invvar = 1 / pow(config['normal_scale'], 2)
        self._normal_const = math.log(2 * math.pi / self._normal_invvar)
        self._seg_overlap = config['seg_overlap']
        with self._enter_variable_scope(check_same_graph=False):
            self._init = Initializer(config)
            self._upd = Updater(config)
            self._dec = Decoder(config)
            self._ln_grad_apc = snt.LayerNorm(axis=[-3, -2, -1], offset=False, scale=False, name='ln_grad_apc')
            self._ln_grad_mask = snt.LayerNorm(axis=[-3, -2, -1], offset=False, scale=False, name='ln_grad_mask')
            self._ln_pixel_ll = snt.LayerNorm(axis=[-3, -2, -1], offset=False, scale=False, name='ln_ll')
            self._ln_pixel_ll_excl = snt.LayerNorm(axis=[-3, -2, -1], offset=False, scale=False, name='ln_ll_exclude')
            self._ln_grad_post_param = snt.LayerNorm(axis=[-1], offset=False, scale=False, name='ln_grad_post_param')

    def _build(self, images, segments, overlaps, num_slots, num_steps, step_wt):
        batch_size = get_batch_size(images)
        idx_range = [*range(num_slots)]
        indices = []
        for idx in range(num_slots):
            indices += idx_range[:idx] + idx_range[idx + 1:]
        gather_indices = tf.constant(indices, dtype=tf.int32, name='gather_indices')
        loc, raw_scale, states = self._init(batch_size, num_slots)
        loss_all_list = []
        for step in range(num_steps):
            with tf.name_scope('step_{}'.format(step + 1)):
                with tf.GradientTape() as tape:
                    tape.watch([loc, raw_scale])
                    apc, mask, log_mask, logits_mask = self._dec(loc, raw_scale, batch_size)
                    raw_pixel_ll = self._get_raw_pixel_ll(images, apc)
                    with tf.name_scope('pixel_ll'):
                        pixel_ll = tf.math.reduce_logsumexp(log_mask + raw_pixel_ll, axis=1)
                    loss_all, loss_nll, loss_kld, loss_grad = self._get_loss_values(pixel_ll, loc, raw_scale)
                with tf.name_scope('gradients'):
                    grad_loc, grad_scale, grad_apc = tape.gradient(loss_grad, [loc, raw_scale, apc])
                upd_in = self._get_upd_in(gather_indices, images, apc, mask, logits_mask, raw_pixel_ll, pixel_ll,
                                          grad_apc, loc, raw_scale, grad_loc, grad_scale, batch_size)
                loc, raw_scale, states = self._upd(loc, raw_scale, upd_in, states, batch_size)
                loss_all_list.append(loss_all)
        with tf.name_scope('loss_opt'):
            loss_opt = tf.expand_dims(step_wt, axis=1) * tf.stack(loss_all_list)
            loss_opt = tf.math.reduce_sum(loss_opt, axis=0) / tf.math.reduce_sum(step_wt)
        losses = {'nll': loss_nll, 'kld': loss_kld, 'opt': loss_opt, 'compare': loss_all}
        with tf.name_scope('outputs'):
            recon = tf.math.reduce_sum(mask * apc, axis=1)
            segment = tf.math.argmax(mask, axis=1)
            mask_oh = tf.one_hot(segment, mask.shape[1], axis=1)
            pres = tf.math.reduce_max(mask_oh, axis=[-3, -2, -1])
        results = {'image': images, 'apc': apc, 'mask': mask, 'pres': pres, 'recon': recon}
        metrics = self._get_metrics(images, segments, overlaps, results, mask_oh, pixel_ll, batch_size)
        return results, metrics, losses

    def _get_raw_pixel_ll(self, images, apc, name='raw_pixel_ll'):
        with tf.name_scope(name):
            sq_diff = tf.math.squared_difference(apc, tf.expand_dims(images, axis=1))
            raw_pixel_ll = -0.5 * (self._normal_const + self._normal_invvar * sq_diff)
            raw_pixel_ll = tf.math.reduce_sum(raw_pixel_ll, axis=-1, keepdims=True)
        return raw_pixel_ll

    @staticmethod
    def _get_loss_values(pixel_ll, loc, raw_scale, eps=1e-10, name='loss_values'):
        with tf.name_scope(name):
            with tf.name_scope('loss_nll'):
                loss_nll = -tf.math.reduce_sum(pixel_ll, axis=[*range(1, pixel_ll.shape.rank)])
            with tf.name_scope('loss_kld'):
                scale = tf.math.softplus(raw_scale)
                var = tf.math.square(scale)
                loss_kld = 0.5 * (tf.math.square(loc) + var - tf.math.log(var + eps) - 1)
                loss_kld = tf.math.reduce_sum(loss_kld, axis=[*range(1, loss_kld.shape.rank)])
            with tf.name_scope('loss_all'):
                loss_all = loss_nll + loss_kld
            with tf.name_scope('loss_grad'):
                loss_grad = tf.math.reduce_sum(loss_all)
        return loss_all, loss_nll, loss_kld, loss_grad

    def _get_upd_in(self, gather_indices, images, apc, mask, logits_mask, raw_pixel_ll, pixel_ll, grad_apc, loc,
                    raw_scale, grad_loc, grad_scale, batch_size, eps=1e-3, name='updater_inputs'):
        def gather_exclude(x):
            x = tf.gather(x, gather_indices, axis=1)
            x = tf.reshape(x, [batch_size, num_slots, num_slots - 1, *x.shape[2:]])
            return x
        def concat_list(x):
            x = tf.concat(x, axis=-1)
            x = tf.reshape(x, [batch_size * x.shape[1], *x.shape[2:]])
            return x
        with tf.name_scope(name):
            num_slots = apc.shape[1]
            with tf.name_scope('in_images'):
                in_images = tf.expand_dims(images * 2 - 1, axis=1)
                in_images = tf.broadcast_to(in_images, [batch_size, num_slots, *images.shape[1:]])
            with tf.name_scope('in_apc'):
                in_apc = apc * 2 - 1
            with tf.name_scope('in_mask'):
                in_mask = mask * 2 - 1
            with tf.name_scope('in_post_mask'):
                post_mask = tf.math.softmax(raw_pixel_ll, axis=1)
                in_post_mask = post_mask * 2 - 1
            with tf.name_scope('in_grad_apc'):
                in_grad_apc = self._ln_grad_apc(grad_apc)
                in_grad_apc = tf.stop_gradient(in_grad_apc)
            with tf.name_scope('in_grad_mask'):
                max_raw_pixel_ll = tf.math.reduce_max(raw_pixel_ll, axis=1, keepdims=True)
                nominator = tf.math.exp(raw_pixel_ll - max_raw_pixel_ll)
                denominator = tf.math.reduce_sum(mask * nominator, axis=1, keepdims=True)
                in_grad_mask = nominator / (denominator + eps)
                in_grad_mask = self._ln_grad_mask(in_grad_mask)
                in_grad_mask = tf.stop_gradient(in_grad_mask)
            with tf.name_scope('in_ll'):
                max_pixel_ll = tf.math.reduce_max(pixel_ll, axis=[-3, -2, -1], keepdims=True)
                in_pixel_ll = tf.math.exp(pixel_ll - max_pixel_ll)
                in_pixel_ll = self._ln_pixel_ll(in_pixel_ll)
                in_pixel_ll = tf.expand_dims(in_pixel_ll, axis=1)
                in_pixel_ll = tf.stop_gradient(in_pixel_ll)
                in_pixel_ll = tf.broadcast_to(in_pixel_ll, [batch_size, num_slots, *in_pixel_ll.shape[2:]])
            with tf.name_scope('in_ll_exclude'):
                raw_pixel_ll_excl = gather_exclude(raw_pixel_ll)
                logits_mask_excl = gather_exclude(logits_mask)
                log_mask_excl = tf.math.log_softmax(logits_mask_excl, axis=2)
                pixel_ll_excl = tf.math.reduce_logsumexp(log_mask_excl + raw_pixel_ll_excl, axis=2)
                max_pixel_ll_excl = tf.math.reduce_max(pixel_ll_excl, axis=[-3, -2, -1], keepdims=True)
                in_pixel_ll_excl = tf.math.exp(pixel_ll_excl - max_pixel_ll_excl)
                in_pixel_ll_excl = self._ln_pixel_ll_excl(in_pixel_ll_excl)
                in_pixel_ll_excl = tf.stop_gradient(in_pixel_ll_excl)
            with tf.name_scope('in_post_param'):
                in_post_param = tf.concat([loc, raw_scale], axis=-1)
            with tf.name_scope('in_grad_post_param'):
                in_grad_post_param = tf.concat([grad_loc, grad_scale], axis=-1)
                in_grad_post_param= self._ln_grad_post_param(in_grad_post_param)
                in_grad_post_param = tf.stop_gradient(in_grad_post_param)
            with tf.name_scope('inputs_1'):
                inputs_1_list = [
                    in_images,
                    in_apc,
                    in_mask,
                    logits_mask,
                    in_post_mask,
                    in_grad_apc,
                    in_grad_mask,
                    in_pixel_ll,
                    in_pixel_ll_excl,
                ]
                inputs_1 = concat_list(inputs_1_list)
            with tf.name_scope('inputs_2'):
                inputs_2_list = [
                    in_grad_post_param,
                    in_post_param,
                ]
                inputs_2 = concat_list(inputs_2_list)
        return inputs_1, inputs_2

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

    def _get_metrics(self, images, segments, overlaps, results, mask_oh, pixel_ll, batch_size, name='metrics'):
        with tf.name_scope(name):
            segments_obj = segments[:, :-1]
            segments_obj_sel = segments_obj if self._seg_overlap else segments_obj * (1 - overlaps)
            with tf.name_scope('ari'):
                ari_all = self._compute_ari(segments_obj_sel, mask_oh, batch_size)
            with tf.name_scope('mse'):
                pixel_mse = tf.math.squared_difference(results['recon'], images)
                mse = tf.math.reduce_mean(pixel_mse, axis=[*range(1, pixel_mse.shape.rank)])
            with tf.name_scope('ll'):
                ll = tf.math.reduce_sum(pixel_ll, axis=[*range(1, pixel_ll.shape.rank)])
            with tf.name_scope('count_acc'):
                pres_true = tf.math.reduce_max(segments_obj, axis=[-3, -2, -1])
                count_true = tf.math.reduce_sum(pres_true, axis=1)
                count_pred = tf.math.reduce_sum(results['pres'], axis=1) - 1
                count_acc = tf.cast(tf.math.equal(count_true, count_pred), dtype=tf.float32)
            metrics = {'ari_all': ari_all, 'mse': mse, 'll': ll, 'count': count_acc}
        return metrics


def get_model(config):
    net = Model(config)
    return net
