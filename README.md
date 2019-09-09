# Multi-GPU TensorFlow Training
This code is provided as a simple example for how to setup _Tower-Based_ multi-GPU training of deep neural networks in Tensorflow 1.x, inspired by the approach found [here](https://jhui.github.io/2017/03/07/TensorFlow-GPU/).

The _tower-based_ approach duplicates the model across each available GPU during initialization, then splits each batch of training data to be fed to each GPU-contained model. Predictions for each sub-batch are produced and the losses are averaged over all GPUs to obtain the batch-specific model gradients, which are then applied to each of the duplicated models (towers) before the next batch is processed.

The key components for implementing a simple example are shown below (full code in [train.py](./train.py)):

```python
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

```
