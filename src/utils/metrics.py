import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.python.ops.gen_math_ops import squared_difference
from tensorflow.python.ops.math_ops import reduce_sum

class SSDNetBoxLoss(losses.loss):
    """Smooth L1 Loss"""
    def __init__(self, delta) -> None:
        super(SSDNetBoxLoss).__init__(reduction = "none", name = "SSDNetBoxLoss")
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        abs_difference = tf.abs(difference)
        squared_difference = abs_difference ** 2
        loss = tf.where(
                        tf.less(abs_difference, self._delta),
                        0.5 * squared_difference, 
                        abs_difference - 0.5
                       )
        return tf.reduce_sum(loss, axis=-1)

class SSDNetClassificationLoss(losses.loss):
    """Focal Loss"""
    def __init__(self, alpha, gamma):
        super(SSDNetClassificationLoss).__init__(
                    reduction = "auto", name="SSDNetClassificationLoss"
                    )
        self._gamma = gamma
        self._alpha = alpha

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels = y_true, logits = y_pred
                        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

class SSDNetLoss(losses.Loss):
    def __init__(self, num_classes=2, alpha=0.25, gamma=2.0, delta=1.0, reduction="auto", name="SSDNetLoss"):
        super(SSDNetLoss).__init__(reduction=reduction, name=name)
        self._clf_loss = SSDNetClassificationLoss(alpha, gamma)
        self._box_loss = SSDNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[..., :4]
        box_predictions = y_pred[..., :4] 
        cls_labels = tf.one_hot(
                             tf.cast(y_true[:, :, 4] - 1, dtype=tf.int32),
                             depth = self._num_classes,
                             dtype = tf.float32
                             )
        cls_predictions = y_pred[..., 4:]

        object_mask = tf.cast(tf.greater(y_true[:, :, 4], 0.0), dtype=tf.float32)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)

        box_loss = tf.where(object_mask == 1.0, box_loss, 0.0)
        normalizer = tf.reduce_sum(object_mask, axis=-1)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        loss = box_loss + clf_loss
        return loss

if __name__ == "__main__":
    pass
