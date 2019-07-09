
import tensorflow as tf
import tensorflow.contrib.slim as slim


'''
    ==================================================================
    Based on the Xception Paper (https://arxiv.org/pdf/1610.02357.pdf)
    ==================================================================
    Xception  
        使用depthwise separable convolutions 替代 Inception module    
'''

def xception(inputs,num_classes,is_training=True,scope='xception'):
    '''
    Xception
    :param inputs: tensor
    :param num_classes:预测的类型数量
    :param is_training:是否训练
    :param scope:
    :return:
    '''
    with tf.variable_scope('Xception') as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.separable_conv2d], depth_multiplier=1), \
             slim.arg_scope([slim.separable_conv2d, slim.conv2d, slim.avg_pool2d],
                            outputs_collections=[end_points_collection]), \
             slim.arg_scope([slim.batch_norm], is_training=is_training):
            # 具体参数可参考论文
            # ===========ENTRY FLOW==============
            # Block 1
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='valid', scope='block1_conv1')
            net = slim.batch_norm(net, scope='block1_bn1')
            net = tf.nn.relu(net, name='block1_relu1')
            net = slim.conv2d(net, 64, [3, 3], padding='valid', scope='block1_conv2')
            net = slim.batch_norm(net, scope='block1_bn2')
            net = tf.nn.relu(net, name='block1_relu2')
            # 跳跃结构上的操作
            residual = slim.conv2d(net, 128, [1, 1], stride=2, scope='block1_res_conv')
            residual = slim.batch_norm(residual, scope='block1_res_bn')

            # Block 2
            net = slim.separable_conv2d(net, 128, [3, 3], scope='block2_dws_conv1')
            net = slim.batch_norm(net, scope='block2_bn1')
            net = tf.nn.relu(net, name='block2_relu1')
            net = slim.separable_conv2d(net, 128, [3, 3], scope='block2_dws_conv2')
            net = slim.batch_norm(net, scope='block2_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block2_max_pool')
            net = tf.add(net, residual, name='block2_add')
            # 跳跃结构上的操作
            residual = slim.conv2d(net, 256, [1, 1], stride=2, scope='block2_res_conv')
            residual = slim.batch_norm(residual, scope='block2_res_bn')

            # Block 3
            net = tf.nn.relu(net, name='block3_relu1')
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv1')
            net = slim.batch_norm(net, scope='block3_bn1')
            net = tf.nn.relu(net, name='block3_relu2')
            net = slim.separable_conv2d(net, 256, [3, 3], scope='block3_dws_conv2')
            net = slim.batch_norm(net, scope='block3_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block3_max_pool')
            net = tf.add(net, residual, name='block3_add')
            # 跳跃结构上的操作
            residual = slim.conv2d(net, 728, [1, 1], stride=2, scope='block3_res_conv')
            residual = slim.batch_norm(residual, scope='block3_res_bn')

            # Block 4
            net = tf.nn.relu(net, name='block4_relu1')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv1')
            net = slim.batch_norm(net, scope='block4_bn1')
            net = tf.nn.relu(net, name='block4_relu2')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block4_dws_conv2')
            net = slim.batch_norm(net, scope='block4_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block4_max_pool')
            net = tf.add(net, residual, name='block4_add')

            # ===========MIDDLE FLOW===============
            for i in range(8):
                block_prefix = 'block%s_' % (str(i + 5))

                residual = net
                net = tf.nn.relu(net, name=block_prefix + 'relu1')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv1')
                net = slim.batch_norm(net, scope=block_prefix + 'bn1')
                net = tf.nn.relu(net, name=block_prefix + 'relu2')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv2')
                net = slim.batch_norm(net, scope=block_prefix + 'bn2')
                net = tf.nn.relu(net, name=block_prefix + 'relu3')
                net = slim.separable_conv2d(net, 728, [3, 3], scope=block_prefix + 'dws_conv3')
                net = slim.batch_norm(net, scope=block_prefix + 'bn3')
                net = tf.add(net, residual, name=block_prefix + 'add')

            # ========EXIT FLOW============
            # 跳跃结构上的操作
            residual = slim.conv2d(net, 1024, [1, 1], stride=2, scope='block12_res_conv')
            residual = slim.batch_norm(residual, scope='block12_res_bn')
            # Exit 操作
            net = tf.nn.relu(net, name='block13_relu1')
            net = slim.separable_conv2d(net, 728, [3, 3], scope='block13_dws_conv1')
            net = slim.batch_norm(net, scope='block13_bn1')
            net = tf.nn.relu(net, name='block13_relu2')
            net = slim.separable_conv2d(net, 1024, [3, 3], scope='block13_dws_conv2')
            net = slim.batch_norm(net, scope='block13_bn2')
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='same', scope='block13_max_pool')
            net = tf.add(net, residual, name='block13_add')

            net = slim.separable_conv2d(net, 1536, [3, 3], scope='block14_dws_conv1')
            net = slim.batch_norm(net, scope='block14_bn1')
            net = tf.nn.relu(net, name='block14_relu1')
            net = slim.separable_conv2d(net, 2048, [3, 3], scope='block14_dws_conv2')
            net = slim.batch_norm(net, scope='block14_bn2')
            net = tf.nn.relu(net, name='block14_relu2')

            net = slim.avg_pool2d(net, [10, 10], scope='block15_avg_pool')
            # Replace FC layer with conv layer instead
            net = slim.conv2d(net, 2048, [1, 1], scope='block15_conv1')
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='block15_conv2')
            logits = tf.squeeze(logits, [1, 2], name='block15_logits')  # Squeeze height and width only     # 删掉1，2
            predictions = slim.softmax(logits, scope='Predictions')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            end_points['Logits'] = logits
            end_points['Predictions'] = predictions

        return logits, end_points

def xception_arg_scope(weight_decay=0.00001,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001):
    '''
    The arg scope for xception model. The weight decay is 1e-5 as seen in the paper.
    INPUTS:
    - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
    - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
    - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.
    OUTPUTS:
    - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
    '''
    # Set weight_decay for weights in conv2d and separable_conv2d layers.
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=None,
                        activation_fn=None):
        # Set parameters for batch_norm. Note: Do not set activation function as it's preset to None already.
        with slim.arg_scope([slim.batch_norm],
                            decay=batch_norm_decay,
                            epsilon=batch_norm_epsilon) as scope:
            return scope
