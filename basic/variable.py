import tensorflow as tf
#Variable 变量的简单运用
state = tf.Variable(0,name='counter')
print(state.name)#变量的名称

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)#把new_value 加载到state

init = tf.initialize_all_variables()#初始化所有的变量，定义了变量值之后必须用到

with tf.Session() as sess:
    sess.run(init) #激活变量
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

