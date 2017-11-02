#coding=utf-8
# author = fukaiyu
#!/usr/bin/env python
from __future__ import division
import math
import numpy as np
def conv(data, kernel, stride=1, type='valid'):
    '''
        卷积函数
        data: 被卷积的图片，类型: list[a, b]
        kernel: 卷积核，类型: list[m, n]
        stride: 卷积步长，类型int，暂时默认为1
        type：卷积类型，类型string
        输出：
        ret: 卷积后图片，类型：list[(a - m + 1) * (b - n + 1)](valid)或者list[a, b](same)
    '''
    if type == "valid":
        a, b = data.shape
        m, n = kernel.shape
        ret = np.zeros([a - m + 1, b - n + 1])
        for i in range(a - m + 1):
            for j in range(b - n + 1):
                temp = data[i:i+m, j:j+n]
                ret[i, j] = np.sum(temp * kernel)
        return ret
    elif type == "same":
        pass

def maxPooling(data, size, type, stride=0):
    '''
        最大池化函数
        data: 被池化的图片，类型list[a, b]
        size: 池化核大小，类型list[m, n]
        type: 池化类型，类型string
        stride: 池化步长，类型int，暂时默认等于size
        输出:
        ret: 池化后的图片，类型：list[ceil(a / m), ceil(b / n)]
        flag: 池化的最大值所在点，类型: list[a, b]
    '''
    if type == 'same':
        m, n = size
        a, b = data.shape
        (y, x) = (int(math.ceil(a / m)), int(math.ceil(b / n)))
        ret = np.empty([y, x])
        flag = np.zeros_like(data)
        for i in range(0, a, m):
            for j in range(0, b, n):
                temp = data[i:i+m, j:j+n]
                y_, x_ = temp.shape
                idx = np.argmax(temp)
                flag[i + idx // x_, j + idx % x_] = 1
                ret[i // m, j // n] = data[i + idx // x_, j + idx % x_]
        return ret, flag

    else:
        pass

def padding(data, num=0, d=1):
    '''
        填充函数
        data: 待被填充的矩阵，类型list[a, b]
        num: 被填充的数，类型int
        d: 被填充的像素数，类型int
        输出：
        ret：填充后的矩阵
    '''
    a, b = data.shape
    ret = np.zeros([a + 2 * d, b + 2 * d])
    ret[d:d+a, d:d+b] = data
    return ret


class ConvolutionalLayer:
    '''
        卷积层
        私有变量
        self.w: 卷积核 类型list[o, i, a, b]
        分别对应：输出通道数，输入通道数，每个卷积核大小为a * b
        self.b: 偏置 类型list[o]
        self.G: 上一层传下来的残差
        self.data: 保存输入数据，用于计算w的梯度 类型list[batch_size, i, a, b]
    '''
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, momentum=0, convType='valid', lr=0.01, name='Conv', editedWeight=None, editedBias=None, Initializer = 'Gaussian', weightDecay = 0.0, L1Regular = 0.0):
        '''
            卷积层初始化
            in_channel: 输入通道数 类型int
            out_channel: 输出通道数 类型int
            kernel_size: 卷积核大小 类型int
            stride: 步长 类型int
            momentum：动量 类型float
            convType：卷积方式 类型 string
            lr: 学习率 类型float
            name: 层名
            editedWeight, editedBias: 是否手工初始化权重和偏置
            deltaW: 梯度动量 类型list[out_channel, in_channel, kernel_size, kernel_size]
            deltab: 梯度动量 类型list[out_channel]
        '''
        self._out = out_channel
        self._in = in_channel
        self.wx = kernel_size
        self.wy = kernel_size
        if editedWeight == None:
            if Initializer == 'Gaussian':
                self.w = np.random.normal(0, 0.1, [out_channel, in_channel, kernel_size, kernel_size])
            elif Initializer == 'Modified Gaussian':
                self.w = np.random.randn(out_channel, in_channel, kernel_size, kernel_size) / np.sqrt(in_channel)
        else:
            self.w = editedWeight
        if editedBias == None:
            self.b = np.zeros(out_channel)
        else:
            self.b = editedBias
        self.lr = lr
        self.weightDecay = weightDecay
        self.L1Regular = L1Regular
        self.stride = stride
        self.momentum = momentum
        self.convType = convType
        self.name = name
        self.G = np.zeros([out_channel, kernel_size, kernel_size])
        self.deltaW = np.zeros([out_channel, in_channel, kernel_size, kernel_size])
        self.deltab = np.zeros([out_channel])

    def forward(self, data):
        '''
            卷积层前向传播
            输入：data，传入的数据，默认为四维张量，类型list[batch_size, channel_num, m, n]
            分别对应为：batch_size: 每个batch的图片数量，每个图片大小为m * n, 且通道数为channel_num
            输出: ret, 类型list[batch_size, out, m, n] 对应same模式， 类型list[batch_size, out, m - self.wx + 1, n - self.wy + 1]， 对应stride为1的valid模式
        '''
        batch_size, channel_num, m, n = data.shape
        self.batch_size = batch_size
        self.data = data
        if self.convType == 'valid':
            ret = np.zeros([batch_size, self._out, m - self.wx + 1, n - self.wy + 1])
        elif self.convType == 'same':
            ret = np.zerps([batch_size, self._out, m, n])
        if data.shape[1] == self.w.shape[1]:
            for batch_num, pics in enumerate(data):
                for out_channel in range(self._out):
                    for index, pic in enumerate(pics):
                        ret[batch_num, out_channel] += conv(pic, self.w[out_channel, index], self.stride, self.convType)

                    ret[batch_num, out_channel] += self.b[out_channel]
            return ret
        else:
            print("参数不匹配")
            raise ValueError

    def backward(self, G):
        '''
            反向传播
            输入: G，下一层的残差，类型list[batch_size, out_channel, m, n]
            输出: H，该层的残差，类型list[batch_size, in_channel, m, n]
        '''
        #print G
        m, n = self.data.shape[2:]
        self.deltab = self.momentum * self.deltab - G.sum(axis=0).sum(axis=(1, 2))
        self.deltaW = self.momentum * self.deltaW
        if self.convType == 'valid':
            for out_channel in range(self._out):
                for in_channel in range(self._in):
                    for y in range(self.wy):
                        for x in range(self.wx):
                            for batch in range(self.batch_size):
                                self.deltaW[out_channel][in_channel][y][x] -= (self.data[batch][in_channel][y:m-self.wy+1+y:self.stride, x:n-self.wx+1+x:self.stride] * G[batch][out_channel]).sum()

        elif self.convType == 'same':
            pass
        a, b = G.shape[2:]
        paddingG = np.zeros([self.batch_size, self._out, a + 2 * (self.wy - 1), b + 2 * (self.wx - 1)])
        for i in range(self.batch_size):
            for j in range(self._out):
                paddingG[i][j] = padding(G[i][j], d = self.wy - 1)
        H = np.zeros_like(self.data)
        for batch in range(self.batch_size):
            for in_channel in range(self._in):
                for out_channel in range(self._out):
                    H[batch][in_channel] += conv(paddingG[batch][out_channel], self.w[out_channel, in_channel, ::-1, ::-1])

        self.w = (1 - self.weightDecay) * self.w +  self.lr * self.deltaW
        self.w[self.w < 0] += self.L1Regular
        self.w[self.w > 0] -= self.L1Regular
        self.b += self.lr * self.deltab
        return H





class MaxPoolingLayer:
    '''
        最大池化层
        私有变量：
        self.size 池化核大小
        self.stride 池化步长
        self.type 池化类型
        self.flag 池化最大点坐标，用于反向传播求得残差矩阵
        self.m = 输出图片的size
        self.n = 输出图片的size
    '''
    def __init__(self, size, stride=0, type='same', name='Maxpooling'):
        self.size = size
        self.stride = stride
        self.type = type
        self.name = name

    def forward(self, data):
        '''
            前向传播
            输入：data 上一层传输的数据，类型list[batch_size, in_channel, m, n]
            输出：ret 池化后的数据，类型list[batch_size, in_channel, ceil(m / size), ceil(n / size)]
        '''

        batch_size, in_channel, m, n = data.shape
        self.m, self.n = (int(math.ceil(m / self.size[0])), int(math.ceil(n / self.size[1])))
        self.flag = np.zeros_like(data)
        ret = np.empty((batch_size, in_channel, self.m, self.n))
        for batch_num, pics in enumerate(data):
            for index, pic in enumerate(pics):
                ret[batch_num, index], self.flag[batch_num, index] = maxPooling(pic, self.size, self.type)
        return ret

    def backward(self, G):
        '''
            反向传播
            输入: G 下一层的残差，类型list[batch_size, in_channel, ceil(m / size), ceil(n / size)]
            输出: H 该层的残差，类型list[batch_size, in_channel, m, n]，最大池化层直接将最大值填入self.flag指定的原位置，其余位置填充0即可
        '''
        H = np.zeros_like(self.flag)
        a, b = self.flag.shape[2:]
        for idx1, batch in enumerate(G):
            for idx2, pic in enumerate(batch):
                for m in range(pic.shape[0]):
                    for n in range(pic.shape[1]):
                        y = m * self.size[0]
                        y_ = (m + 1) * self.size[0] if (m + 1) * self.size[0] <= a else a
                        x = n * self.size[1]
                        x_ = (n + 1) * self.size[1] if (n + 1) * self.size[1] <= b else b
                        H[idx1, idx2, y:y_, x:x_] = self.flag[idx1, idx2, y:y_, x:x_] * pic[m, n]
        return H



class FullConnectedLayer:
    '''
        全连接层
        私有变量:
        self.w: 权重，类型list[in_channel, out_channel]
        self.b: 偏置，类型list[out_channel]
        self.data: 输入的数据，反向传播时求当层权重的梯度用
        self.deltab: 上一次计算得到的权重的梯度，在使用动量法的情况下使用，类型list[in_channel, out_channel]
        self.deltaW: 上一次计算得到的权重的梯度，在使用动量法得情况下使用，类型list[out_channel]
    '''
    def __init__(self, in_channel, out_channel, lr=0.01, momentum=0, name='FullConnect', editedWeight=None, editedBias=None, Initializer='Gaussian', weightDecay=0.0):
        '''
            in_channel: 输入结点数，类型int
            out_channel: 输出结点数 类型int
            lr: 学习率 类型float
            momentum: 动量 类型float 暂未实现
            name: 名称
            editedWeight, editedBias: 是否手工初始化权重和偏置
        '''
        self.lr = lr
        self.momentum = momentum
        self.name = name
        self.in_channel = in_channel
        self.out_channel = out_channel
        if editedWeight == None:
            if Initializer == 'Gaussian':
                self.w = np.random.normal(0, 0.1, [in_channel, out_channel])
            elif Initializer == 'Modified Gaussian':
                self.w = np.random.randn(in_channel, out_channel) / np.sqrt(in_channel)
        else:
            self.w = editedWeight
        if editedBias == None:
            self.b = np.zeros(out_channel)
        else:
            self.b = editedBias
        self.deltaW = np.zeros([in_channel, out_channel])
        self.deltab = np.zeros([out_channel])
        self.weightDecay = weightDecay

    def forward(self, data):
        '''
            前向传播
            输入：data 平铺后输入的数据，类型list[batch_size, in_channel]
            输出: ret 全连接层计算后的数据，类型list[batch_size, out_channel]
        '''
        self.batch_size, in_channel = data.shape
        self.data = data
        if in_channel == self.in_channel:
            ret = np.dot(data, self.w) + self.b
            return ret
        else:
            print("参数不匹配")
            raise ValueError

    def backward(self, G):
        '''
            反向传播
            输入：G 下一层的残差，类型list[batch_size, out_channel]
            输出：H 该层的残差，通过上一层的残差对该层的权重求导即可得到,类型list[batch_size, in_channel]
        '''
        self.deltab = self.momentum * self.deltab - np.sum(G, axis = 0)
        self.deltaW = self.momentum * self.deltaW - np.dot(self.data.T, G)
        H = np.zeros([self.batch_size, self.in_channel])
        for index, in_channel in enumerate(self.w):
            H[:,index] += np.dot(G, in_channel.T) # list[batch_size, 1]
        self.w = (1 - self.weightDecay) * self.w + self.lr * self.deltaW
        self.b += self.lr * self.deltab
        return H



class SigmoidLayer:
    '''
        sigmoid 层
        私有变量：
        self.data：前向传播时接收到的上一层的数据，存储用于反向传播求残差
    '''
    def __init__(self, name='sigmoid'):
        self.name = name

    def forward(self, data):
        '''
            前向传播
            输入：data 上一层传输的数据，类型list[batch_size, in_channel]
            输出：ret 激活后的数据，类型不变
        '''
        self.data = data
        ret = 1.0 / 1.0 + np.exp(-data)
        return ret

    def backward(self, G):
        '''
            反向传播
            输入：G  下一层的残差，类型list[batch_size, in_channel]
            输出: H  该层的残差，类型list[batch_size, in_channel]
        '''
        H = (1.0 / 1.0 + np.exp(-self.data)) * (1.0 - 1.0 / 1.0 + np.exp(-self.data))
        return G * H



class ReluLayer:
    '''
        Relu层
        私有变量：
        self.data: 前向传播时接收到的上一层的数据，存储用于反向传播求残差
    '''
    def __init__(self, name='Relu'):
        self.name = name

    def forward(self, data):
        '''
            前向传播
            输入：data 上一层传输的数据，类型list[batch_size, in_channel, m, n]针对卷积层 或者 list[batch_size, in_channel]针对fc层
            输出: ret 整流后的数据，类型不变，其中小于0的元素置为0
        '''
        self.data = data
        ret = data.copy()
        ret[ret < 0] = 0
        return ret

    def backward(self, G):
        '''
            反向传播
            输入：G 下一层的残差，类型list[batch_size, in_channel, m, n]针对池化层 或者 list[batch_size, in_channel]针对fc层
            输出：G 该层的残差，类型list[batch_size, in_channel, m, n]针对卷积层 或者 list[batch_size, in_channel]针对fc层
        '''
        G[self.data < 0] = 0
        return G

class SoftmaxLayer:
    '''
        softmax层
        私有变量：
        self.out: 输出，存储起来以便反向传播计算残差
    '''
    def __init__(self, name='Softmax'):
        self.name = name

    def forward(self, data):
        '''
            前向传播
            输入: data, 输入的数据，类型list[batch_size, in_channel]
            输出: self.out, 归一化后的数据，类型list[batch_size, in_channel]
        '''
        batch_size, in_channel = data.shape
        exp = np.exp(data)
        Sum = np.sum(exp, axis=1).reshape(batch_size, -1)
        self.out = exp / Sum
        return self.out

    def backward(self, G):
        '''
           反向传播
            输入：G，下一层传入的残差，softmax作为最后一层，直接接受损失函数的残差，在使用交叉熵作为损失函数情况下残差为该层的输出与最终的交叉熵，类型list[batch_size, in_channel]
            输出：残差 self.out - G，类型list[batch_size, in_channel]
        '''
        return self.out - G


class FlattenLayer:
    '''
        平铺层, 按照tensorflow的标准，将其平铺
        私有变量：
        self.batch_size: batch数量
        self.channel: 通道数
        self.m self.n: 图片大小
    '''
    def __init__(self, name='flatten'):
        self.name = name

    def forward(self, data):
        '''
            前向传播
            输入: data，上一层的数据，类型list[batch_size, in_channel, m, n]
            输出: ret, 平铺后的数据，类型list[batch_size, in_channel * m * n]
        '''
        self.batch_size, self.channel, self.m, self.n = data.shape
        ret = np.zeros([self.batch_size, self.channel * self.m * self.n])
        for i in range(data.shape[0]):
            ret[i] = np.stack(data[i].reshape(data[i].shape[0], -1), axis=1).reshape(-1)

        return ret

    def backward(self, G):
        '''
            反向传播
            输入：G，下一层的残差，类型list[batch_size, in_channel * m * n]
            输出: ret, 恢复结构后的数据，类型list[batch_size, inchnnel, m, n]
        '''
        ret = np.zeros([self.batch_size, self.channel, self.m, self.n])
        for i in range(G.shape[0]):
            for j in range(self.channel):
                ret[i][j] = G[i][j::self.channel].reshape(self.m, self.n)


        return ret

class CrossEntropy:
    '''
        交叉熵损失函数
        目前只接受softmax层作为输出层的反向传播计算
    '''
    def forward(self, data, label):
        '''
            前向传播
            输入：data, softmax层归一化后的向量，类型list[batch_size, out_channel]
            输出：output, 计算得到的output和希望得到的结果y之间的损失函数
            类型list[batch_size, int]
        '''
        indexs = [np.argmax(y) for y in label]
        output = np.sum(np.nan_to_num([-np.log(data[i][indexs[i]]) for i in range(data.shape[0])]))
        return output

    def backward(self, data, label):
        '''
            反向传播
            输入：label，训练集的类别，类型list[batch_size, out_channel]
            输出：label，训练集的类别，类型list[batch_size, out_channel]
        '''
        return label

class QuadraticCost:
    '''
        平方损失函数
    '''
    def forward(self, data, label):
        '''
            前向传播
            输入：data，类型list[batch_size, out_channel]
                  label，类型list[batch_size, out_channel]
            输出：output, 类型list[batch_size, int]
        '''
        return np.sum((data  - label) ** 2, axis=1)

    def backward(self, data, label):
        '''
            反向传播
            输入：data, 类型list[batch_size, out_channel]
                ：label, 类型list[batchs_size, out_channel]
            输出：output, 类型list[bathc_size, out_channel]
        '''
        return data - label

class Net:
    '''
        生成模型
    '''
    def __init__(self):
        self.layers=[]

    def addLayer(self, layer):
        self.layers.append(layer)

    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        '''
            训练函数
        '''
        train_num = trainData.shape[0]
        for it in range(iteration):
            print 'iter=' + str(it)
            for batch in range(0, train_num, batch_size):
                print 'batch=' + str(batch)
                if batch + batch_size < train_num:
                    self.train_inner(trainData[batch : batch + batch_size], trainLabel[batch : batch + batch_size])
                else:
                    self.train_inner(trainData[batch : train_num], trainLabel[batch : train_num])
            print 'eval=' + str(self.eval(validData, validLabel))

    def train_inner(self, data, label):
        '''
            内部训练函数
        '''
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num - 1):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data

        print self.layers[lay_num - 1].forward(in_data, label)
        G = self.layers[lay_num - 1].backward(in_data, label)

        for i in range(lay_num - 2, -1, -1):
            G = self.layers[i].backward(G)

    def eval(self, data, label):
        '''
            评估函数
        '''
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num - 1):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)
        label_idx = np.argmax(label, axis=1)
        return np.sum(out_idx == label_idx) / float(out_idx.shape[0])

    def save(self, filename):
        '''
            存储权重和模型结构
            输入：文件名 类型string
        '''
        pass
def load(filename):
    pass


