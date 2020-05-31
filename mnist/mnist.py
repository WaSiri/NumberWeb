
#1.0成功
import tensorflow as tf
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#导入mnist数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(mnist.train.labels.shape)


#CNN权重
# 参考https://blog.csdn.net/justgg/article/details/94362621
# 即tf.truncated_normal( 截断正态分布
# 	shape,               shape为常量(向量空间)的形状，如shape=[2,3,4]为X=2,Y=3,Z=4的矩阵
# 	mean=None,           正态分布均值（默认为0）
# 	stddev=None,         正态分布标准差（默认为1） 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。
# 	dtype=None,          dtype为输出的数据类型，如tf.float32等
# 	seed=None,           随机数种子，为整型，设置后每次生成的随机数都一样。
# 	name=None            该函数操作的名字
# )
# tf.random_normal与tf.truncated_normal在于tf.random_normal会在整个正态分布空间内进行取值，而不是（μ-2σ，μ+2σ）内
# 关于seed,可用以下语句增加理解：
# i=0
# while(i<3):
#     tn1 = tf.truncated_normal([6],stddev=1,seed=1)
#     sess = tf.Session()
#     print(sess.run(tn1))
#     i+=1

# j=0
# while(j<3):
#     tn2 = tf.truncated_normal([6],stddev=1)
#     sess = tf.Session()
#     print(sess.run(tn2))
#     j+=1

def weight_variable(shape):  #用对称破坏的小噪声初始化权重
    initial =  tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#偏置参数
#参考https://blog.csdn.net/csdn_jiayu/article/details/82155224
# tf.constant(               创建常量
#     value,                 value可为数字和数组，
#     dtype=None,            dtype为常量的数据类型，如tf.float32等，
#     shape=None,            shape为常量(向量空间)的形状，如shape=[2,3,4]为X=2,Y=3,Z=4的矩阵
#     name='Const',          name为该常量的名字，string类型
#     verify_shape=False     verify_shape默认为False，如果修改为True的话表示检查value的形状与shape是否相符，如果不符会报错。
# )

def bias_variable(shape):  #将偏置参数初始化为小的正数，以避免死神经元
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#卷积
# 参考https://blog.csdn.net/mao_xiao_feng/article/details/78004522
# 参考https://blog.csdn.net/zuolixiangfisher/article/details/80528989
# 参考https://blog.csdn.net/qq_30934313/article/details/86626050
# tf.nn.conv2d (				卷积层
# 	input, 						输入，即卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。
# 	filter, 					卷积核参数，要求为一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
# 	strides,                    卷积步长，是一个长度为4的一维向量，[ 1, strides(height), strides(width), 1]，第一位和最后一位一般是1
# 	padding, 					string类型，值为“SAME” 和 “VALID”，表示是否考虑边界填充。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑    ！！不同的值会造成不同的输出公式，详见https://blog.csdn.net/alxe_made/article/details/80834305
# 	use_cudnn_on_gpu, 	        bool类型，是否使用cudnn加速，默认为true（gpu真的贵，穷学生买不起）	
# 	name=None                   该函数操作的名字
# )
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# #最大池化
# 参考https://blog.csdn.net/m0_37586991/article/details/84575325
# 参考https://blog.csdn.net/mao_xiao_feng/article/details/53453926
# tf.nn.max_pool(   			池化层
#     input,   					输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, in_height, in_width, in_channels]这样的shape 其中，in_height为卷积map的out_height, in_width卷积后map的out_width, in_channels=filter的out_channel
#     ksize,  					池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为1
#     strides,  				池化滑动步长，是一个长度为4的一维向量，[ 1, strides, strides, 1]，第一位和最后一位一般是1
#     padding,					string类型，值为“SAME” 和 “VALID”，表示是否考虑边界填充。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
#     name=None        			该函数操作的名字
# )

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 初始化输入X
# 参考https://blog.csdn.net/kdongyi/article/details/82343712
# tensorflow.compat.v1  因为tensorflow2.x的版本不再支持placeholder，因此其中提供了tensorflow.compat.v1代码包来兼容原有1.x的代码，可以做到几乎不加修改的运行。
# 在tf1.x版本中，我们必须将图表分为两个阶段：
# 构建一个描述您要执行的计算的计算图。这个阶段实际上不执行任何计算;它只是建立了计算的符号表示。该阶段通常将定义一个或多个表示计算图输入的“占位符”（placeholder）对象。
# 多次运行计算图。 每次运行图形时（例如，对于一个梯度下降步骤），您将指定要计算的图形的哪些部分，并传递一个“feed_dict”字典，该字典将给出具体值为图中的任何“占位符”。
# tf.placeholder(    此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
# 	dtype, 			 数据类型。常用的是tf.float32,tf.float64等数值类型
# 	shape=None, 	 数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定），因为MNIST数据为24*24*1图像，所以为784
# 	name=None        该函数操作的名字
# )
x = tf.placeholder(tf.float32, shape=[None,784])
# 初始化输出Y
# 因为MNIST为[0,9]共十个分类
y_ = tf.placeholder(tf.float32,shape=[None,10])

#第一个卷积层
#创建卷积核W_conv1,表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32
W_conv1 =  weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# 参考https://blog.csdn.net/apengpengpeng/article/details/80579454
# 参考https://blog.csdn.net/m0_37592397/article/details/78695318
# tf.reshape(          改变张量形式
#     tensor,          tensor类型，即placeholder所处理得到的
#     shape,           输出的张量形式
#     name=None        该函数操作的名字
# )
#把输入x(二维张量,shape为[None, 784])变成4维的x_image，x_image的shape是[None,28,28,1]
#-1表示自动推测这个维度的size,即传给None
x_image = tf.reshape(x,[-1,28,28,1])
#relu激化和池化
# tf.nn.relu，使用relu激活函数
# 参考https://blog.csdn.net/tyhj_sf/article/details/79932893查看激活函数的作用
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,32]
h_pool1 = max_pool_2x2(h_conv1)

#第二个卷积层
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
#该层拥有1024个神经元（神经元个数可在0~4000间调参）
#W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# 将第2层的输出reshape成[batch, 7*7*64]的张量
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 参考https://www.cnblogs.com/AlvinSui/p/8987707.html
# tf.matmul(  矩阵相乘x*y
# 	x,    	  矩阵x    
# 	y 		  矩阵y
# )
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

#Dropout减少过拟合
keep_prob = tf.placeholder(tf.float32)
#参考https://blog.csdn.net/huahuazhu/article/details/73649389
#参考https://blog.csdn.net/yangfengling1023/article/details/82911306
# tf.nn.dropout(
#     x,					输入,为tensor类型
#     keep_prob,			float类型，每个元素被保留下来的概率，设置神经元被选中的概率,在初始化时keep_prob只是一个占位符
#     noise_shape=None,     一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。
#     seed=None,			随机数种子，为整型，设置后每次生成的随机数都一样。
#     name=None     		该函数操作的名字
# )
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


#读出层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2


#用交叉熵作为代价函数
# 参考https://www.jianshu.com/p/cf235861311b
# 参考https://blog.csdn.net/mieleizhi0522/article/details/80200126
# 参考https://blog.csdn.net/zwqjoy/article/details/78952087
cross_entropy = tf.reduce_mean(
	# 参考https://blog.csdn.net/mao_xiao_feng/article/details/53382790（评论第一条有大用）
	# 参考https://blog.csdn.net/qq_35203425/article/details/79773459
	# tf.nn.softmax_cross_entropy_with_logits(
	# 	logits, 		神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes，需和标签大小一致
	# 	labels, 		实际的标签
	# 	name=None   	该函数操作的名字
	# )
    tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = y_conv)
    )

# 引入tf.train.AdamOptimizer().minimize进行梯度下降，学习率为0.0001
# 函数内容详见https://blog.csdn.net/TeFuirnever/article/details/88933368
# 		   及https://blog.csdn.net/lomodays207/article/details/84027365
# 		   及https://blog.csdn.net/polyhedronx/article/details/93405760及官方文档
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#评估函数
# 参考https://www.jianshu.com/p/1aec70f77b36
# 参考https://blog.csdn.net/kdongyi/article/details/82390394
# tf.argmax(						返回最大的那个数值所在的下标
#     input,						输入矩阵
#     axis=None,				    axis可被设置为0或1，分别表示0：按列计算，1：行计算
#     name=None,					设置函数名称
#     dimension=None,
#     output_type=tf.int64			返回数据类型
#     )
# 参考https://blog.csdn.net/ustbbsy/article/details/79564529
# tf.equal(				判断输入x,y是否相等
# 	x, 				    输入x
# 	y,    				输入y
# 	name=None  			设置函数名称
# 	)

y_true =  tf.argmax(y_, 1)
y_pre = tf.argmax(y_conv,1)
correct_prediction = tf.equal(y_pre ,y_true)
# 参考https://blog.csdn.net/dcrmg/article/details/79747814
# tf.cast(         执行 tensorflow 中张量数据类型转换
#   input,         待转换的数据（张量）   
# 	dtype, 		   目标数据类型
# 	name=None      设置函数名称
# 	)
# 参考https://blog.csdn.net/dcrmg/article/details/79797826
# tf.reduce_mean(      				计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
# 	input_tensor,					输入张量
#     axis=None,					指定的轴，如果不指定，则计算所有元素的均值;
#     keep_dims=False, 				是否降维度，设置为True，输出的结果保持输入tensor的形状，设置为False，输出结果会降低维度;
#     name=None,					设置函数名称
#     )
#计算正确预测项的比例，即准确率
#因为tf.equal返回的是布尔值，使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

#运行
F1_arr = []
with tf.Session() as sess:
	# 参考https://blog.csdn.net/qq_37285386/article/details/89054090
	# 参考https://blog.csdn.net/surserrr/article/details/89421925
	# sess.run(tf.global_variables_initializer())    初始化全局所有变量
    sess.run(tf.global_variables_initializer())

    for i in range(4000):
    	#train.next_batch(50)   每次随机从训练集中抓取50幅图像
    	# 参考https://blog.csdn.net/qq_33254870/article/details/81390897
    	# 参考https://blog.csdn.net/qq_41140351/article/details/92710525
        batch = mnist.train.next_batch(50)  
        if i % 40 == 0:
        	#feed_dict 和 placeholder的相生相灭
        	# 参考https://blog.csdn.net/lcczzu/article/details/91416211
        	# 参考https://www.cnblogs.com/itboys/p/8858172.html

        	# accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 }) = sess.run(accuracy,feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 })  
        	# 参考https://blog.csdn.net/zxyhhjs2017/article/details/82349535
        	# 参考https://blog.csdn.net/qq_31150463/article/details/84561478
        	# 参考https://segmentfault.com/a/1190000015287066?utm_source=tag-newest
            #参考https://blog.csdn.net/jiaoyangwm/article/details/79248535
            train_accuracy = accuracy.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 })
            y_true_arr = y_true.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 })
            y_pre_arr = y_pre.eval(feed_dict = {x:batch[0], y_:batch[1], keep_prob:1.0 })
            f1 = f1_score(y_true_arr, y_pre_arr, average='weighted')
            F1_arr.append(f1)
            print('step %d, f1 = %g' % (i, f1))
        train_step.run(feed_dict = {x:batch[0], y_:batch[1], keep_prob:0.5 })
        
        

xrange = list(range(100))
plt.title('Result Analysis')
plt.plot(xrange, F1_arr, "-.", color='green', label='F1')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('F1')
plt.show()

    # 保存模型
    #saver.save(sess, 'D:/CSU/program/Python/Practice/Vs2019/my_mnist/my_mnist/model.ckpt')

# print('test accuracy %g' % accuracy.eval(
#     feed_dict = {
#         x: mnist.test.images,
#         y_: mnist.test.lables,
#         keep.prob:1.0 }))   调整参数所用的代码块

'''
#2.0
import numpy as np
import tensorflow as tf

#导入mnist
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
features = mnist.train.images #返回np.array

#输入层
INPUT = tf.reshape(features, [-1, 28, 28, 1])

#卷积层1
CONV1 = tf.layers.conv2d(
    inputs = INPUT,
    filters = 32,
    kernel_size = [5,5],
    padding = "same",
    activation = tf.nn.relu)

#池化层1
POOL1 = tf.layers.max_pooling2d(
    inputs = CONV1,
    pool_size = [2,1],
    strides = 2)

#卷积层2
CONV2 = tf.layers.conv2d(
    inputs = POOL1,
    filters = 64,
    kernel_size = [5,5],
    padding = "same",
    activation = tf.nn.relu)

#池化层2
POOL2 = tf.layers.max_pooling2d(
    inputs = CONV2,
    pool_size = [2,1],
    strides = 2)

#全连接层（1024个神经元）
POOL2_FLATTENED = tf.reshape(POOL2,[-1,7*7*64])
FC1 = tf.layers.dense(inputs = POOL2_FLATTENED, unitf = 1024, activation = tf.nn.relu)

#dropout层
DROPOUT = tf.layers.dropout(
    inputs=FC1,
    rate=0.5,
    training=mode == tf.estimator.ModeKeys.TRAIN)
FC2 = tf.layers.dense(inputs = DROPOUT, units=10)

#交叉熵损失函数
onehot_labels = tf.one_hot(indices = tf.cast(labels,tf.int32),depth = 10)
loss = tf.losses.softmax_cross_entropy(onehot_labels,logits = FC2)

#配置训练操作
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(loss=loss,
                              global_step = tf.train.get_global_step())
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x":features},
    y = lables,
    batch_size = 100,
    num_epochs = None,
    shuffle = True)
minst_classifier = tf.estimator.EstimatorSpec(
    mode = mode,
    loss = loss,
    train_op = train_op)
mnist_classifier.train(input_fn = train_input_fn, steps = 20000)

#3成功
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


input = tf.placeholder(tf.float32,[None,784])
input_image = tf.reshape(input,[-1,28,28,1])

y = tf.placeholder(tf.float32,[None,10])

# input 代表输入，filter 代表卷积核
def conv2d(input,filter):
    return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')
# 池化层
def max_pool(input):
    return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# 初始化卷积核或者是权重数组的值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

# 初始化bias的值
def bias_variable(shape):
    return tf.Variable(tf.zeros(shape))

#[filter_height, filter_width, in_channels, out_channels]
#定义了卷积核
filter = [3,3,1,32]

filter_conv1 = weight_variable(filter)
b_conv1 = bias_variable([32])
# 创建卷积层，进行卷积操作，并通过Relu激活，然后池化
h_conv1 = tf.nn.relu(conv2d(input_image,filter_conv1)+b_conv1)
h_pool1 = max_pool(h_conv1)

h_flat = tf.reshape(h_pool1,[-1,14*14*32])

W_fc1 = weight_variable([14*14*32,768])
b_fc1 = bias_variable([768])
h_fc1 = tf.matmul(h_flat,W_fc1) + b_fc1

W_fc2 = weight_variable([768,10])
b_fc2 = bias_variable([10])

y_hat = tf.matmul(h_fc1,W_fc2) + b_fc2



cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_hat ))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        batch_x,batch_y = mnist.train.next_batch(50)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={input:batch_x,y:batch_y})
            print("step %d,train accuracy %g " %(i,train_accuracy))

        train_step.run(feed_dict={input:batch_x,y:batch_y})

        # sess.run(train_step,feed_dict={x:batch_x,y:batch_y})

    print("test accuracy %g " % accuracy.eval(feed_dict={input:mnist.test.images,y:mnist.test.labels}))
'''
