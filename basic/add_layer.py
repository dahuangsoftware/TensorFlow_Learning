import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#添加神经层
def add_layer(input,insize,outsize,actication_function=None):
    Weights = tf.Variable(tf.random_normal([insize,outsize]))
    biases = tf.Variable(tf.zeros([1,outsize])+0.1)
    wx_plus_b =  tf.matmul(input,Weights)+biases
    if actication_function is None:
        output = wx_plus_b
    else:
        output = actication_function(wx_plus_b)
    return output

#产生一些数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#定义输入的placeholder
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])
#添加隐藏层
l1 = add_layer(xs,1,10,actication_function=tf.nn.relu)
#添加输出层
predition = add_layer(l1,10,1,actication_function=None)

#计算loss的差别
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#参数为学习效率

#初始化所有的变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 ==0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        predition_value = sess.run(predition,feed_dict={xs:x_data})
        #plot预测值
        lines = ax.plot(x_data, predition_value, 'r-', lw=5)
        plt.pause(0.2)
        plt.show()