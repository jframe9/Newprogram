import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets(r'D:\Computerwork\Programs\Python\Newprogram\demo\Mnist_data', one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个占位 placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
W1 = tf.Variable(tf.zeros([784, 10]))
b1 = tf.Variable(tf.zeros([10]))
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# L1_drop = tf.nn.dropout(L1, keep_prob=1.0)
prediction = tf.nn.softmax(tf.matmul(x, W1) + b1)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))
# 损失函数用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法来实现最优
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing accuracy " + str(acc))
