import math

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import (
    placeholder,
    variable_scope,
    get_collection,
    ConfigProto,
    GPUOptions,
    Session,
    GraphKeys
)
from tensorflow.compat.v1.train import (
    exponential_decay,
    MomentumOptimizer,
)

default_device = '/gpu:0'

def model(examples, is_training=True, output_channels=1, depth=3, nfilters=16):
    """Example model"""
    output = tf.layers.conv2d(examples, kernel_size=(1,1), filters=output_channels, use_bias=True, name='1x1conv2d')
    return tf.nn.relu(output)

def get_available_gpus():
    """Get the list of usable GPUs"""
    with Session(config=ConfigProto(gpu_options=GPUOptions(per_process_gpu_memory_fraction=0.01, allow_growth=True))) as sess:
        tf.logging.set_verbosity(tf.logging.WARN)
        local_device_protos = device_lib.list_local_devices()

        devs =  [x.name for x in local_device_protos if x.device_type=='GPU']
    return devs

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for tt, (g, v) in enumerate(grad_and_vars):
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class RandomBatchConstructor():
    """Construct batches randomly from a set of example (input, label) paired tensors """
    def __init__(self, inputs, labels):
        """ note: currently only handles 2D inputs. 3D extension should be simple

        Args:
            inputs (np.ndarray[N,H,W,C])
            labels (np.ndarray[N,H,W,C])
        """
        self.inputs = inputs
        self.labels = labels
        self.randorder = np.arange(self.inputs.shape[0])
        self.mark = 0
        self.initialized = False # lazy loading of index

    def reset(self):
        """re-init the array of available example indices"""
        self.initialized = True
        np.random.shuffle(self.randorder)
        self.mark = 0

    def make_batch(self, batch_size):
        """construct batch in NHWC format where the first axis (N) is constucted of random examples drawn from list of unused examples"""
        if not self.initialized:
            self.reset()
        remaining = len(self.randorder) - self.mark
        _batch_size = min(remaining, batch_size)
        if remaining < _batch_size:
            raise RuntimeError('There are not enough examples ({:d}) to fill the requested batch ({:d})'.format(remaining, _batch_size))
        selection = self.randorder[self.mark:self.mark+_batch_size]
        self.mark += _batch_size
        return (self.inputs[selection,], self.labels[selection,])

class ModelExample(object):
    def __init__(self, sess, lr_init=1e-4, weighted=True, input_c_dim=2, depth=3, nfilters=16):
        with tf.device('/cpu:0'):
            self.sess = sess
            self.devices = get_available_gpus()
            print('running on devices: {!s}'.format(self.devices))

            # model arguments
            self.input_c_dim = input_c_dim

            # build model
            self.labels = placeholder(tf.float32, [None, None, None, 1], name="labels") # low var dose
            self.inputs = placeholder(tf.float32, [None, None, None, self.input_c_dim], name="inputs")  # high var dose + geometry
            self.predictions = None # defined below

            self.is_training = placeholder(tf.bool, name='is_training')
            self.global_step = placeholder(tf.int32, name='global_step')
            self.epoch_size  = placeholder(tf.int32, name='epoch_size')
            self.lr = exponential_decay(lr_init,
                                        self.global_step,
                                        self.epoch_size,
                                        0.9842,
                                        staircase=True,
                                        name='lr_exponential_decay')

            optimizer = MomentumOptimizer(self.lr, 0.9, name='MomentumOptimizer')

            nsamples = tf.shape(self.inputs)[0]
            ndevices_ = len(self.devices)
            ndevices = tf.constant(ndevices_)

            tower_grads = []
            tower_loss  = []
            with variable_scope('model') as modelscope:
                for ii, dev in enumerate(self.devices):
                    nper = tf.cast(tf.ceil(nsamples/ndevices), tf.int32)
                    if ii == ndevices-1:
                        nper = nsamples - (ndevices-1)*nper

                    with tf.name_scope('tower_{:d}'.format(ii)):
                        with tf.device(dev):
                            b_in = self.inputs[nper*ii:nper*(ii+1)]
                            b_lab = self.labels[nper*ii:nper*(ii+1)]
                            b_pred = model(b_in, is_training=self.is_training, depth=depth, nfilters=nfilters)

                            # force towers to share parameters (stored on gpu:0)
                            modelscope.reuse_variables()

                            loss = tf.reduce_mean(tf.losses.mean_squared_error(b_lab, b_pred))
                            grad = optimizer.compute_gradients(loss)
                            tower_grads.append(grad)
                            tower_loss.append(loss)

                with tf.device(default_device):
                    self.loss = tf.reduce_mean(tower_loss)
                    # order of predictions in multi-gpu mode is not guaranteed so we need to compute in single-gpu mode for evaluation.
                    self.predictions = model(self.inputs, is_training=self.is_training, depth=depth, nfilters=nfilters)

            # update model params - gradient step
            grads = average_gradients(tower_grads)
            update_ops = get_collection(GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.apply_gradients(grads)

            init = tf.global_variables_initializer()
            self.sess.run(init)

    def train(self, train_in, train_label, batch_size, numEpoch):
        iteration = 0

        tf.get_default_graph().finalize() # produce error rather than allow memory leaks
        batchcon = RandomBatchConstructor(train_in, train_label)
        num_batches = math.floor(len(train_in) / batch_size)
        for epoch in range(numEpoch):
            batchcon.reset()
            for (batch_in, batch_label) in batchcon.iter_batches(batch_size):
                if len(batch_in) < batch_size:
                    # don't allow undersized batches
                    continue

                _, loss = self.sess.run(
                    [self.train_op, self.loss], feed_dict={
                        self.inputs: batch_in,
                        self.labels: batch_label,
                        self.is_training: True,
                        self.global_step: iteration,
                        self.epoch_size: num_batches,
                        }
                    )
                iteration += 1


if __name__ == '__main__':
    sess = Session()
    model = ModelExample(sess)
    inputs2d = tf.random.normal((100, 100, 100))
    labels   = tf.random.normal((100, 1))
    batchsize = 100
    numEpoch = 10
    model.train(inputs2d, labels, batchsize, numEpoch)
