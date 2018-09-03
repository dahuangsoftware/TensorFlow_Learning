#session 的打开方式

import tensorflow as tf

matrix1 = tf.constant([[3,3]])#常量
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2) #矩阵乘法

#方法一
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

#方法2,sess最后会自动关闭
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

