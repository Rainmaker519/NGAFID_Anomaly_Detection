/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Random;

import data.ImageDataSet;
import data.CIFARDataSet;
import data.MNISTDataSet;

import network.ConvolutionalNode;
import network.Edge;
import network.PoolingEdge;
import network.ConvolutionalEdge;
import network.ConvolutionalNeuralNetwork;
import network.LossFunction;
import network.NeuralNetworkException;

import network.ActivationType;
import network.NodeType;

import util.Log;



public class PA33TestCNNs {
    public static final boolean checkGradients = true;

    public static void main(String[] arguments) {
        //test LeNet-5
        try {
            //test for MNIST
            MNISTDataSet mnistTrain = new MNISTDataSet("./datasets/train-images-idx3-ubyte", "./datasets/train-labels-idx1-ubyte", 60000);
            mnistTrain.resize(1000);

            boolean useDropout = false;
            double inputDropoutRate = 0.0;
            double hiddenDropoutRate = 0.0;
            boolean useBatchNormalization = true;
            double alpha = 0.9;

            PA31TestCNNs.testCNN(mnistTrain, "mnist_small_no_pool", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(mnistTrain, "mnist_small", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(mnistTrain, "mnist_lenet5", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

            PA31TestCNNs.testCNN(mnistTrain, "mnist_small_no_pool", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(mnistTrain, "mnist_small", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(mnistTrain, "mnist_lenet5", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

            CIFARDataSet cifarTrain = new CIFARDataSet(new String[]{"./datasets/cifar-10-batches-bin/data_batch_1.bin", "./datasets/cifar-10-batches-bin/data_batch_2.bin", "./datasets/cifar-10-batches-bin/data_batch_3.bin", "./datasets/cifar-10-batches-bin/data_batch_4.bin", "./datasets/cifar-10-batches-bin/data_batch_5.bin"}, 10000);
            cifarTrain.resize(5000);

            PA31TestCNNs.testCNN(cifarTrain, "cifar_small_no_pool", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(cifarTrain, "cifar_small", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(cifarTrain, "cifar_lenet5", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

            PA31TestCNNs.testCNN(cifarTrain, "cifar_small_no_pool", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(cifarTrain, "cifar_small", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            PA31TestCNNs.testCNN(cifarTrain, "cifar_lenet5", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);


        } catch (NeuralNetworkException e) {
            System.err.println("CNN test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        } 
    }
}

