# author = fukaiyu
#!/usr/bin/env python
# coding=utf-8
import numpy as np
import math
import cnn 
import input_data
data_sets = input_data.read_data_sets('../../data_sets/MNIST_data/', one_hot=True)
net = cnn.Net()
net.addLayer(cnn.ConvolutionalLayer(1, 6, 5, lr=0.01, weightDecay=0.001))
net.addLayer(cnn.ReluLayer())
net.addLayer(cnn.MaxPoolingLayer([2, 2]))
net.addLayer(cnn.ConvolutionalLayer(6, 16, 5, lr=0.01, weightDecay=0.001))
net.addLayer(cnn.ReluLayer())
net.addLayer(cnn.MaxPoolingLayer([2, 2]))
net.addLayer(cnn.FlattenLayer())
net.addLayer(cnn.FullConnectedLayer(16 * 4 * 4, 84, lr=0.01, weightDecay=0.001))
net.addLayer(cnn.ReluLayer())
net.addLayer(cnn.FullConnectedLayer(84, 10, lr=0.01, weightDecay=0.001))
net.addLayer(cnn.SoftmaxLayer())
net.addLayer(cnn.CrossEntropy())


image = data_sets.train.images
label = data_sets.train.labels
net.train(image[:1000], label[:1000], image[1000:1100], label[1000:1100], 10, 10)
