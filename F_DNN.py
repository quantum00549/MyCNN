import tensorflow as tf
import DNN_Essay
from tensorflow.examples.tutorials.mnist import input_data

def full_connected(input_tensor, hidden_layer, regularizer=None, dropout=None, reuse=False):
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
            tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(0.00001)(weights))
    return layer


mnist = input_data.read_data_sets('./mnist/data',one_hot=True)
data_size = mnist.train.num_examples
test_data_size = mnist.test.num_examples
STEPS = 310
learning_rate_base = 0.7
learning_rate_decay = 0.95
batch_size = int(0.9*data_size)
hidden_layer = [28, 10]
regularizer = tf.contrib.layers.l1_regularizer(0.0001)


X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels
image_size = mnist.train.images.shape[1]


# regularizer = tf.contrib.layers.l2_regularizer(0.001)
x = tf.placeholder(tf.float32, [None, image_size], name='x_input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')
y = full_connected(x, hidden_layer)
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y)
    )
tf.add_to_collection('losses', cost)
loss = tf.add_n(tf.get_collection('losses'))
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,100,learning_rate_decay,staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
total_cross_entropy = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_prob = tf.nn.softmax(y)
    correct_prediction = tf.equal(tf.argmax(y_prob, 1), tf.argmax(Y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for i in range(STEPS):
        print(i)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        print(sess.run(loss,feed_dict={x:X_train, y_:Y_train}))
        train_accuracy = accuracy.eval({x: X_test, y_: Y_test})
        print(train_accuracy)
    v = tf.get_collection('losses')
