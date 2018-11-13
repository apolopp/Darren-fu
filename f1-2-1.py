# import tensorflow as tf
#
# a=tf.constant(2)
# b=tf.Variable(3,name='counter')
#
# produce=tf.add(a,b)
#
# updata=tf.assign(b,produce)
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3):
#         print(sess.run(produce))

import tensorflow as tf
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

product=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(product,feed_dict={input1:[2],input2:[11]}))
