import tensorflow as tf
import numpy as np
#生成随机序列
x_data = np.random.rand(100).astype(np.float32)
y_data =  x_data*0.1 + 0.3

#生成tensorflow结构
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))#预测值与真实值的差别
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)#减少误差

init = tf.initialize_all_variables()
#tensorflow结构搭建完毕

sess = tf.Session()
sess.run(init) #激活、非常重要

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))