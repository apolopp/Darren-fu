import tensorflow as tf
matrix1=tf.constant([[1,2]])
matrix2=tf.constant([[3],[4]])

product=tf.matmul(matrix1,matrix2)

# sess=tf.Session()
# print(sess.run(product))
# sess.close()
with tf.Session() as sess:
    re=sess.run(product)
    print(re)