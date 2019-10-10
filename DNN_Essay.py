#!/usr/bin/env python
# coding: utf-8


# -*- coding:utf-8 -*-
"""
@author:          Bin.Chen
@date:            2019/07/08
@software:       Jupyter Notebook
@Environment :   python3.6
@Description:    CNN框架，力求网络结构完全可控
"""


import numpy as np
import tensorflow as tf


def full_connected(input_tensor, hidden_layer, regularizer=None, dropout=None, reuse=False):
    """
    全连接层的计算
    :input_tensor:           输入数据，矩阵，形如[[1,2],[3,4]]表示矩阵第一行为[1,2]，第二行为[3,4]
    :param hidden_layer: 隐藏层结构，列表，例：[3,4]表示第一个隐藏层有3个节点，第二个隐藏层有4个节点，以此类推，可自行增减，
                                 所谓隐藏层，不包含输入层，但包含输出层，调整参数时注意最后一层节点个数
    :return:                    经计算后的输出
    """
    layer = input_tensor
    for i in range(1,len(hidden_layer)+1):
        with tf.variable_scope('layer{}'.format(i),reuse=reuse):
            weights = tf.get_variable(
                'weights',shape=[layer.shape[1],hidden_layer[i-1]],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1,seed=1)
            )
            if regularizer != None:
                tf.add_to_collection('losses',regularizer(weights))
            biases = tf.get_variable(
                'bias',shape=[hidden_layer[i-1]],initializer=tf.constant_initializer(0.1)
            )
            layer = tf.nn.tanh(
                tf.matmul(layer,weights)+biases
            )
            if dropout != None:
                layer = tf.nn.dropout(layer,dropout[0],noise_shape=dropout[1])
    return layer


def conv(input_tensor,structure,reuse=False):
    """
    卷积和池化层的计算
    :input_tensor:           输入数据
    :param structure:      形如
                                                        structure = {
                                                    1:{
                                                        'conv':{'filter':[5,5,3,16],'stride':[1,1,1,1],'padding':'SAME'},
                                                        'pool':{'filter':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
                                                    }
                                                }
                                    直接将其复制粘贴，1表示第一层卷积层结构，如需要多层卷积，加2，3。。。照搬格式即可，
                                    conv对应的filter为卷积过滤器尺寸、当前层深度、过滤器深度，当前层深度初值为图形数据的深度，
                                    如果有多层卷积，注意structure中当前层深度等于前一层的过滤器深度,
                                    stride首尾为1，不可更改，中间俩表示长宽维度上的步长，
                                    padding表示是否使用全零填充，SAME或者VALID
                                    pool对应的filter为池化过滤器尺寸，首尾必须为1，stride意义同上，
                                    如果没有池化层，删去'pool'对应的字典即可
    :return:                    经计算后的输出
    """
    clayer = input_tensor
    for i in range(1,len(structure)+1):
        with tf.variable_scope('clayer{}'.format(i),reuse=reuse):
            filter_weight = tf.get_variable(
                'weights',structure[i]['conv']['filter'],initializer=tf.truncated_normal_initializer(mean=0,stddev=0.1,seed=1)
            )
            biases = tf.get_variable(
                'biases',[structure[i]['conv']['filter'][3]],initializer=tf.constant_initializer(0.1)
            )
            conv = tf.nn.conv2d(
                clayer,filter_weight,structure[i]['conv']['stride'],padding=structure[i]['conv']['padding']
            )
            bias = tf.nn.bias_add(conv,biases)
            activated_conv = tf.nn.tanh(bias)
            if 'pool' in structure[i]:
                pool = tf.nn.max_pool(
                    activated_conv,ksize=structure[i]['pool']['filter'],strides=structure[i]['pool']['stride'],
                    padding=structure[i]['conv']['padding']
                )
                clayer = pool
            else:
                clayer = activated_conv
    return clayer


# In[ ]:


"""
以下是函数调用demo，所有需要输入的参数均在本cell中，部分功能暂时可以在函数中简单修改，如自定义正则化类
"""
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data',one_hot=True)
data_size = mnist.train.num_examples
# 获得数据集数量
test_data_size = mnist.test.num_examples
image_size = int(math.sqrt(mnist.train.images.shape[1]))
# shape返回一个元祖，(数据数量，数据值)
channel = 1
X_train = np.reshape(mnist.train.images,(data_size,image_size,image_size,channel))
Y_train = mnist.train.labels
X_test = np.reshape(mnist.test.images,(test_data_size,image_size,image_size,channel))
Y_test = mnist.test.labels
# 一个标签是一个列表

tol = 0.0001
# 终止条件
batch_size = int(0.9*data_size)
# 随机梯度下降的一个batch大小，设为1*data_size即为不使用随机梯度下降
STEPS = 22
# 迭代轮数上限
learning_rate_base = 0.7
# 初始学习率
learning_rate_decay = 0.95
# 学习率衰退速度，设为1即为不适用指数衰减法
stair_num = 100
# 梯形衰退参数，每过stair_num轮迭代，指数衰减一次
dropout = [0.2,None]
# dropout参数设置，第一个参数表示权重变为0的概率，第二个参数可以使得矩阵的一部分全为0，是一个列表，不需要此功能则为None
# 例如：[0.5,None]，也可以是[0.5,[3,1]]
# 如果不用dropout功能，dropout = None即可
optimizer = 'Adam'
# 优化方法选择，可选：Adam, GradientDescent,Momentum，如有需要，可以自行编写优化函数
hidden_layer = [84,10]
# 全连接层的隐藏层结构，参数说明见上文
conv_structure = {
    1:{
        'conv':{'filter':[5,5,1,6],'stride':[1,1,1,1],'padding':'SAME'},
        'pool':{'filter':[1,2,2,1],'stride':[1,2,2,1],'padding':'SAME'}
    },
    2:{
        'conv':{'filter':[5,5,6,16],'stride':[1,1,1,1],'padding':'SAME'},
        'pool':{'filter':[1,2,2,1],'stride':[1,2,2,1],'padding':'SAME'}
    }
}
# 卷积层结构，参数说明见上文
regularizer = tf.contrib.layers.l2_regularizer(0.001)
# 正则化参数，如需使用L1正则化，将函数名中的2改为1即可


x = tf.placeholder(tf.float32,[None,image_size,image_size,channel],name='x_input')
y_ = tf.placeholder(tf.float32,[None,10],name='y_input')
clayer = conv(x,structure=conv_structure)
pool_shape = clayer.get_shape().as_list()
nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
reshaped = tf.reshape(clayer,[-1,nodes])
# 因为要使用随机梯度下降，一个batch内数据量不固定，参数-1表示由程序确定第一个维度大小，
# 原本用None也行，但是None会报错，应当是新版本有所改变
y = full_connected(reshaped,hidden_layer,regularizer,dropout)
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
)
tf.add_to_collection('losses',cost)
loss = tf.add_n(tf.get_collection('losses'))
global_step = tf.Variable(0,trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,10,learning_rate_decay,staircase=True)
# staircase参数为True表示学习率梯形下降，每过一定轮数迭代乘以learning_rate_decay
if optimizer == 'Adam':
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
elif optimizer == 'GradientDescent':
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
elif optimizer == 'Momentum':
    train_step = tf.train.MomentumOptimizer(learning_rate).minimize(loss,global_step=global_step)
total_cross_entropy = []
# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_prob = tf.nn.softmax(y)
    correct_prediction = tf.equal(tf.argmax(y_prob, 1), tf.argmax(Y_test, 1))
    # tf.equal返回的是长度为batch_size的一维数组，内容是布尔值true或者false
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(STEPS):
        # start = (i*batch_size)%data_size
        # end = min(start+batch_size, data_size)

        # tf.cast(data, dtype)的作用是将data的类型转换为dtype类型,比如这里，
        # 把bool类型的correct_prediction转换成tf.float32，就实现了true或者false变成了0或者1的转换
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, (batch_size, image_size, image_size, channel))
        sess.run(train_step,feed_dict={x:batch_x, y_:batch_y})
        total_cross_entropy.append(sess.run(loss,feed_dict={x:X_train, y_:Y_train}))
        if i%3 ==0:
            print(i, total_cross_entropy[i])
            train_accuracy = accuracy.eval({x: X_test, y_: Y_test})
            print(train_accuracy)
        # if i > 0:
        #     if abs(total_cross_entropy[i]-total_cross_entropy[i-1]) <= tol:
        #         # saver.save(sess,'./saved_model/model/model.ckpt')
        #         break
    # saver.save(sess,'./saved_model/model/model.skpt')


# saver = tf.train.Saver()
#     with tf.Session() as sess:
        # saver.restore(sess,'./saved_model/model/model.ckpt')
    # y_prob = sess.run(tf.nn.softmax(y),feed_dict={x:X_test})
    # y_label = sess.run(tf.argmax(y_prob,1))
    # hit = 0
    # for i in range(test_data_size):
        # print(list(y_prob[i]).index(max(y_prob[i])))
        # print(y_prob[i])
        # print(list(Y_test[i]).index(max(Y_test[i])))
        # print(Y_test[i])
        # if list(y_prob[i]).index(max(y_prob[i])) == list(Y_test[i]).index(max(Y_test[i])):
        #     hit += 1
    # accuracy = hit/test_data_size


