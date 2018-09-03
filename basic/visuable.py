import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#添加神经层
def add_layer(input,insize,outsize,actication_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([insize,outsize]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,outsize])+0.1,name='b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b =  tf.add(tf.matmul(input,Weights),biases)
        if actication_function is None:
            output = wx_plus_b
        else:
            output = actication_function(wx_plus_b,)
        return output

#产生一些数据
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#定义输入的placeholder
with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
#添加隐藏层
l1 = add_layer(xs,1,10,actication_function=tf.nn.relu)#有一个激活函数是线性函数变弯
#添加输出层
predition = add_layer(l1,10,1,actication_function=None)

#计算loss的差别
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#参数为学习效率

sess = tf.Session()
wirter = tf.summary.FileWriter("logs/",sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

# direct to the local dir and run this in terminal:
# $ tensorboard --logdir=logs