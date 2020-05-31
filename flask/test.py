import tensorflow as tf

#  CNN权重
#  用对称破坏的小噪声初始化权重
def weight_variable(shape):
    initial =  tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#  偏置参数
def bias_variable(shape):  # 将偏置参数初始化为小的正数，以避免死神经元
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#  卷积核
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#  池化层（采用最大池化）
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def predict(image):
    print("hello world")
    result = image[0]
    print(image.shape)
    #  初始化输入X
    #  输入大小为28*28
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None,784])
    #  初始化输出Y
    #  因为MNIST为[0,9]共十个分类
    y_ = tf.placeholder(tf.float32,shape=[None,10])

    # 第一个卷积层
    # 创建卷积核W_conv1,表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
    W_conv1 =  weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x,[-1,28,28,1])
    # relu激化和池化
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    # h_pool1的输出即为第一层网络输出，shape为[batch,14,14,32]
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二个卷积层
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    # 该层拥有1024个神经元（神经元个数可在0~4000间调参）
    # W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    #  将第2层的输出reshape成[batch, 7*7*64]的张量
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

    # Dropout减少过拟合
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


    # 读出层
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

    # cross_entropy = tf.reduce_mean(
    #     tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv)
    #     )
    #
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    # 运行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./models/model.ckpt") #使用模型，参数和之前的代码保持一致
        prediction=tf.argmax(y_conv,1)
        predint = prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
        print('识别结果:')
        print(predint[0])
    return predint[0]

