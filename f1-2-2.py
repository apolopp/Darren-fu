# import tensorflow as tf
# import numpy as np
# def add_layer(inputs,in_size,out_size,activation_function=None):
#
#     weights=tf.Variable(tf.random_normal([in_size,out_size]))
#     biases=tf.Variable(tf.zeros([1,out_size])+0.1)
#     wx_plus_b=tf.matmul(inputs,weights)+biases
#
#
#     if activation_function is None:
#         result=wx_plus_b
#     else:
#         result=activation_function(wx_plus_b)
#
#     return result
#
#
# x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
# noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
# y_data=np.square(x_data)-0.5+noise
#
# xs=tf.placeholder(tf.float32,[None,1])
# ys=tf.placeholder(tf.float32,[None,1])
#
# l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
# prediction=add_layer(l1,10,1,activation_function=None)
#
# loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#
# train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# init=tf.global_variables_initializer()
# sess=tf.Session()
# sess.run(init)
#
# for i in range(10000):
#     sess.run(train,feed_dict={xs:x_data,ys:y_data})
#     if i%50==0:
#         print(i,sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(input,int_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([int_size,out_size]))
    Biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(input,Weights)+Biases
    if activation_function is None:
        output=Wx_plus_b
    else:
        output=activation_function(Wx_plus_b)
    return output

x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]#一行变成一列
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))#矩阵变形
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()


for i in range(10000):
    sess.run(train,feed_dict={xs:x_data,ys:y_data})
    if i%50==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data,ys:y_data})
        lines=ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(1)