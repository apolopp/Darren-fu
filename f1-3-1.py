import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs,in_size, out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    Biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights)+Biases
    if activation_function is None:
        result = Wx_plus_b
    else:
        result = activation_function(Wx_plus_b)

    return result


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))#equal，argmax判断最大数值的下标是否一致
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#cast把对应数值转换成对应类型
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:

        print(compute_accuracy(mnist.test.images, mnist.test.labels))