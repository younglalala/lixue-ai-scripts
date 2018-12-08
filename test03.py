import tensorflow as tf
import numpy as np
#成绩数据20个
xx = [140., 139., 120., 128., 133., 144., 145., 128., 130., 135.,110.,137.,121.,100.,108.,102.,123.,116.,118.,122.]   #数学成绩
yy = [111., 120., 100., 98., 95., 101., 114., 112., 109., 108.,99.,105.,102.,115.,108.,116.,107.,111.,110.,120.]    #预测语文成绩

def get_batch(x,y,batch_size):
    lens=len(x)
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        idx=np.random.randint(lens)
        batch_x.append(x[idx])
        batch_y.append( y[idx])
    return np.array(batch_x),np.array(batch_y)


class Predict:
    def __init__(self):
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.y_=tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.w=tf.Variable(tf.random_normal([1,1],dtype=tf.float32))
        self.b=tf.Variable(tf.zeros([1]))
    def forward(self):
        self.out= tf.matmul(self.x,self.w) + self.b
    def backward(self):
        self.loss=tf.reduce_mean(tf.reduce_sum(tf.square(self.y_ - self.out)))
        self.opt=tf.train.AdamOptimizer(0.1).minimize(self.loss)
if __name__=='__main__':
    predict=Predict()
    predict.forward()
    predict.backward()
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for j in range(3000):
            x,y=get_batch(xx, yy, 2)
            x=x.reshape([-1,1])
            y=y.reshape([-1,1])
            _,loss,out=sess.run([predict.opt,predict.loss,predict.out],feed_dict={predict.x:x,predict.y_:y})
            # print(loss)
            # print(out,'out')
            # print(y,'lable')
        predict_num=145.       #输入数学成绩
        predict_out=sess.run(predict.out,feed_dict={predict.x:np.array([[predict_num]])})
        print('数学成绩为:{}，预测这位同学的语文成绩为:{}'.format(predict_num,predict_out[0][0]))    #

