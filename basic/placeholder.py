import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

#使用placeholder在运行的时候再赋值
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7],input2:[8]}))#传入input的值，以feed_dict的形式传入