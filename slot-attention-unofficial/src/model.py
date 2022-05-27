import tensorflow as tf
from network import Encoder, Decoder
from utils import batch_mean, get_checkpoint


class Model(tf.keras.Model):

    def __init__(self, config):
        super(Model, self).__init__()
        # Hyperparameters
        self.seg_overlap = config['seg_overlap']
        # Neural networks
        self.enc = Encoder(config)
        self.dec = Decoder(config)

    def call(self, data, num_slots, require_extra):
        image, segment, overlap = self.convert_data(data)
        slots = self.enc(image, num_slots)
        apc, mask, logits_mask = self.dec(slots)
        recon = tf.reduce_sum(apc * mask, axis=1)
        seg = tf.argmax(mask, axis=1, output_type=tf.int32)
        mask_oh = tf.one_hot(seg, mask.shape[1], axis=1)
        pres = tf.reduce_max(mask_oh, axis=[-3, -2, -1])
        outputs = {'recon': recon, 'apc': apc, 'mask': mask, 'pres': pres, 'mask_oh': mask_oh, 'seg': seg}
        if require_extra:
            results = {'image': image, 'logits_mask': logits_mask, **outputs}
        else:
            results = {}
        metrics = self.get_metrics(image, segment, overlap, outputs)
        loss = batch_mean(tf.square(recon - image) * 4)
        losses = {'opt': loss}
        losses['compare'] = losses['opt']
        return results, metrics, losses

    @staticmethod
    def convert_data(data):
        return data['image'], data['segment'], data['overlap']

    @staticmethod
    def get_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.shape.rank > 1:
                x = tf.reduce_sum(x, axis=[*range(1, x.shape.rank)])
            return x
        num_pixels = tf.reduce_sum(mask_true, axis=[*range(1, mask_true.shape.rank)])
        mask_true = tf.reshape(
            mask_true, [mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[2] * mask_true.shape[3]])
        mask_pred = tf.reshape(
            mask_pred, [mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[2] * mask_pred.shape[3]])
        mat = tf.reduce_sum(mask_true * mask_pred, axis=-1)
        sum_row = tf.reduce_sum(mat, axis=1)
        sum_col = tf.reduce_sum(mat, axis=2)
        comb_mat = comb2(mat)
        comb_row = comb2(sum_row)
        comb_col = comb2(sum_col)
        comb_num = comb2(num_pixels)
        comb_prod = (comb_row * comb_col) / comb_num
        comb_mean = 0.5 * (comb_row + comb_col)
        diff = comb_mean - comb_prod
        score = (comb_mat - comb_prod) / diff
        invalid = tf.math.logical_or(tf.equal(comb_num, 0), tf.equal(diff, 0))
        score = tf.where(invalid, tf.ones_like(score), score)
        return score

    def get_metrics(self, image, segment, overlap, outputs):
        mask_oh = outputs['mask_oh']
        recon = outputs['recon']
        pres = outputs['pres']
        # ARI
        segment_obj = segment[:, :-1]
        segment_obj_sel = segment_obj if self.seg_overlap else segment_obj * (1 - overlap)
        ari = self.get_ari(segment_obj_sel, mask_oh)
        # MSE
        mse = batch_mean(tf.square(recon - image))
        # Count
        pres_true = tf.reduce_max(segment, axis=[-3, -2, -1])
        count_true = tf.reduce_sum(pres_true, axis=1)
        count_pred = tf.reduce_sum(pres, axis=1)
        count_acc = tf.cast(tf.equal(count_true, count_pred), tf.float32)
        metrics = {'ari': ari, 'mse': mse, 'count': count_acc}
        return metrics


class CustomizedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, config):
        super(CustomizedSchedule, self).__init__()
        self.learning_rate = config['lr']
        self.decay_rate = config['lr_decay']
        self.decay_steps = config['decay_steps']
        self.warmup_steps = config['warmup_steps']

    def __call__(self, step):
        lr = tf.constant(self.learning_rate, dtype=tf.float32)
        dr = tf.constant(self.decay_rate, dtype=tf.float32)
        step = tf.cast(step, dtype=tf.float32)
        decay_ratio = step / self.decay_steps
        decay_coef = tf.pow(dr, decay_ratio)
        warmup_ratio = step / self.warmup_steps
        warmup_coef = tf.minimum(warmup_ratio, 1)
        scaled_lr = decay_coef * warmup_coef * lr
        return scaled_lr

    def get_config(self):
        key_list = ['learning_rate', 'decay_rate', 'decay_steps', 'warmup_steps']
        config = {key: getattr(self, key) for key in key_list}
        return config


def get_model(strategy, config):
    with strategy.scope():
        net = Model(config)
        lr_schedule = CustomizedSchedule(config)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-08)
        checkpoint = get_checkpoint(config, net, optimizer)
    return net, optimizer, checkpoint
