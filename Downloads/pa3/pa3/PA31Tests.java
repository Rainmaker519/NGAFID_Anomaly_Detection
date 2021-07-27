/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 1 - Part 1.
 *
 */
import java.util.Arrays;
import java.util.Random;

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



public class PA31Tests {
    public static final int NUMBER_REPEATS = 1;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values
    public static final boolean checkGradients = false;

    public static void main(String[] arguments) {
        //test to make sure we're calculating the average and standard deviation
        //of pixels for the 1 channel MNIST data
        testMNISTValidation();

        //test to make sure we're calculating the average and standard deviation
        //of pixels for the 3 channel CIFAR data
        testCIFARValidation();


        //test convolutional nodes
        try {
            //test a convolutional node with to make sure
            //a basic forward and backward pass is working
            //first with a batch size of 1 and 1 channel
            int batchSize = 1;
            int sizeZ = 1;
            int sizeY = 10;
            int sizeX = 10;
            testConvolutionalNode(10 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(32 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(43 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);

            //then with a batch size of 10 and 1 channel
            batchSize = 10;
            testConvolutionalNode(110 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(332 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(443 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);

            //then with a batch size of 1 and 3 channels
            batchSize = 1;
            sizeZ = 3;
            testConvolutionalNode(112 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(334 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(445 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);

            //then with a batch size of 10 and 3 channels
            batchSize = 10;
            sizeZ = 3;
            testConvolutionalNode(11220 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(33442 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            testConvolutionalNode(44553 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
        } catch (NeuralNetworkException e) {
            System.err.println("ConvolutionalNode test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }

        //test convolutional edges
        try {
            int inputPadding = 0;
            int batchSize = 1;
            int inputSizeZ = 1;
            int inputSizeY = 10;
            int inputSizeX = 10;
            int outputPadding = 0;
            int outputSizeZ = 1;
            int outputSizeY = 5;
            int outputSizeX = 5;

            testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            //try with different paddings
            inputPadding = 2;
            batchSize = 1;
            inputSizeZ = 1;
            inputSizeY = 10;
            inputSizeX = 10;
            outputPadding = 2;
            outputSizeZ = 1;
            outputSizeY = 5;
            outputSizeX = 5;

            testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            //try with different paddings
            inputPadding = 3;
            batchSize = 1;
            inputSizeZ = 1;
            inputSizeY = 10;
            inputSizeX = 10;
            outputPadding = 2;
            outputSizeZ = 1;
            outputSizeY = 5;
            outputSizeX = 5;

            testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            //try with different paddings
            inputPadding = 1;
            batchSize = 1;
            inputSizeZ = 1;
            inputSizeY = 10;
            inputSizeX = 10;
            outputPadding = 2;
            outputSizeZ = 1;
            outputSizeY = 5;
            outputSizeX = 5;

            testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

        } catch (NeuralNetworkException e) {
            System.err.println("ConvolutionalEdge test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }

        //test pooling edges
        try {
            int batchSize = 1;
            int inputPadding = 0;
            int inputSizeZ = 1;
            int inputSizeY = 4;
            int inputSizeX = 4;
            int outputPadding = 0;
            int outputSizeZ = 1;
            int outputSizeY = 2;
            int outputSizeX = 2;
            int poolSize = 2;
            int stride = 2;

            testPoolingEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);

            inputSizeY = 6;
            inputSizeX = 6;
            outputSizeY = 2;
            outputSizeX = 2;
            poolSize = 3;
            stride = 3;
            testPoolingEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(220 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(330 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);

            inputSizeY = 5;
            inputSizeX = 5;
            outputSizeY = 2;
            outputSizeX = 2;
            poolSize = 3;
            stride = 2;
            testPoolingEdge(1103 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(2204 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(3305 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);

            inputSizeY = 11;
            inputSizeX = 11;
            outputSizeY = 5;
            outputSizeX = 5;
            poolSize = 3;
            stride = 2;
            testPoolingEdge(1103 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(2204 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            testPoolingEdge(3305 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
        } catch (NeuralNetworkException e) {
            System.err.println("PoolingEdge test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void testMNISTValidation() {
        MNISTDataSet mnistTrain = new MNISTDataSet("C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\train-images-idx3-ubyte", "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\train-labels-idx1-ubyte", 60000);

        double[] avgs = mnistTrain.getChannelAvgs();
        double[] stdDevs = mnistTrain.getChannelStdDevs(avgs);

        Log.info("MNIST data set had the following channel avgs: " + Arrays.toString(avgs));
        Log.info("MNIST data set had the following channel std devs: " + Arrays.toString(stdDevs));

        //don't uncomment these as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray1d(avgs, "pa31_mnist_avgs", 0);
        if (generatingTestValues) TestValues.writeArray1d(stdDevs, "pa31_mnist_stddevs", 0);

        try {
            Log.info("Checking normalization avgs for MNIST");
            TestValues.testArray1d(avgs, TestValues.readArray1d("pa31_mnist_avgs", 0), "pa31_mnist_avgs", 0);
            Log.info("normalization avgs were correct for MNIST.");

            Log.info("Checking normalization std devs for MNIST");
            TestValues.testArray1d(stdDevs, TestValues.readArray1d("pa31_mnist_stddevs", 0), "pa31_mnist_stddevs", 0);
            Log.info("normalization std devs were correct for MNIST.");

        } catch (NeuralNetworkException e) {
            Log.fatal("Normalization not correctly implemented, calcualted the wrong normalization avg or std dev values: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
    public static void testCIFARValidation() {
        CIFARDataSet cifarTrain = new CIFARDataSet(new String[]{"C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\cifar-10-batches-bin/data_batch_1.bin", "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\cifar-10-batches-bin/data_batch_2.bin", "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\cifar-10-batches-bin/data_batch_3.bin", "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\cifar-10-batches-bin/data_batch_4.bin", "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\datasets\\cifar-10-batches-bin/data_batch_5.bin"}, 10000);

        double[] avgs = cifarTrain.getChannelAvgs();
        double[] stdDevs = cifarTrain.getChannelStdDevs(avgs);

        Log.info("CIFAR data set had the following channel avgs: " + Arrays.toString(avgs));
        Log.info("CIFAR data set had the following channel std devs: " + Arrays.toString(stdDevs));

        //don't uncomment these as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray1d(avgs, "pa31_cifar_avgs", 0);
        if (generatingTestValues) TestValues.writeArray1d(stdDevs, "pa31_cifar_stddevs", 0);

        try {
            Log.info("Checking normalization avgs for CIFAR");
            TestValues.testArray1d(avgs, TestValues.readArray1d("pa31_cifar_avgs", 0), "pa31_cifar_avgs", 0);
            Log.info("normalization avgs were correct for CIFAR.");

            Log.info("Checking normalization std devs for CIFAR");
            TestValues.testArray1d(stdDevs, TestValues.readArray1d("pa31_cifar_stddevs", 0), "pa31_cifar_stddevs", 0);
            Log.info("normalization std devs were correct for CIFAR.");

        } catch (NeuralNetworkException e) {
            Log.fatal("Normalization not correctly implemented, calcualted the wrong normalization avg or std dev values: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void testConvolutionalNode(int seed, ActivationType activationType, int batchSize, int sizeZ, int sizeY, int sizeX, boolean checkGradients, boolean useDropout, double dropoutRate, boolean useBatchNormalization, double alpha) throws NeuralNetworkException {
        int layer = 1;
        int number = 1;

        ConvolutionalNode node = new ConvolutionalNode(layer, number, NodeType.HIDDEN, activationType, 0, batchSize, sizeZ, sizeY, sizeX, useDropout, dropoutRate, useBatchNormalization, alpha);

        Random generator = new Random(seed);
        double[] weights = new double[node.getNumberWeights()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (generator.nextDouble() * 0.10) - 0.05;
        }

        node.setWeights(0, weights);

        //save the input values so we can use them to calculate the numeric gradient for the CNN node
        double[][][][] inputValues = new double[batchSize][sizeZ][sizeY][sizeX];
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        inputValues[i][z][y][x] = (generator.nextDouble() * 11.0) - 3.0;


                        node.inputValues[i][z][y][x] = inputValues[i][z][y][x];
                    }
                }
            }
        }

        node.propagateForward(true);

        //do not uncomment these as they will overwrite the correct values I've generated for the test
        String extraName = "_convolutional_node_" + activationType + "_" + batchSize + "_" + sizeZ + "_" + sizeY + "_" + sizeX + "_" + useDropout + "_" + dropoutRate + "_" + useBatchNormalization  + "_" + alpha;
        String extraText = "seed: " + seed + ", activationType: " + activationType + ", batchSize: " + batchSize + ", sizeZ: " + sizeZ + ", sizeY: " + sizeY + ", sizeX: " + sizeX + ", dropout: " + useDropout + ", dropoutRate: " + dropoutRate + ", batchNorm: " + useBatchNormalization + ", alpha: " + alpha;

        if (generatingTestValues) TestValues.writeArray4d(node.inputValues, "inputValues" + extraName, seed);
        if (generatingTestValues) TestValues.writeArray4d(node.outputValues, "outputValues" + extraName, seed);

        Log.info("Checking inputValues for " + extraText);
        TestValues.testArray4d(node.inputValues, TestValues.readArray4d("inputValues" + extraName, seed), "inputValues" + extraName, seed);

        Log.info("Checking outputValues for " + extraText);
        TestValues.testArray4d(node.outputValues, TestValues.readArray4d("outputValues" + extraName, seed), "outputValues" + extraName, seed);

        if (!checkGradients) return; //don't test the gradients for PA3-1

        double[] numericGradient = getConvolutionalNodeNumericGradient(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX);
        Log.info("numeric gradient: " + Arrays.toString(numericGradient));

        //don't uncomment this as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray1d(numericGradient, "numeric_gradient" + extraName, seed);

        Log.info("Checking numeric_gradient for " + extraText);
        TestValues.testArray1d(numericGradient, TestValues.readArray1d("numeric_gradient" + extraName, seed), "numeric_gradient" + extraName, seed);

        //getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX);

        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        node.delta[i][z][y][x] = 1;
                    }
                }
            }
        }

        node.propagateBackward();

        double[] deltas = new double[node.getNumberWeights()];
        node.getDeltas(0, deltas);

        for (int j = 0; j < deltas.length; j++) {
            Log.debug("node.deltas[" + j + "]: " + deltas[j]);
        }

        Log.info("checking to see if numeric gradient and backprop deltas are close enough for convolutional node.");
        if (!BasicTests.gradientsCloseEnough(numericGradient, deltas)) {
            throw new NeuralNetworkException("backprop vs numeric gradient check failed for " + extraText);
        }
    }

    public static double getConvolutionalNodeOutput(ConvolutionalNode node, double[][][][] inputs, double[] weights, int batchSize, int sizeZ, int sizeY, int sizeX, double[][][][] deltaMods) {
        node.reset();
        node.setWeights(0, weights);

        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        node.inputValues[i][z][y][x] = inputs[i][z][y][x];
                    }
                }
            }
        }

        node.propagateForward(true);

        double outputSum = 0.0;
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        //Log.info("outputValues[" + i + "][" + z + "][" + y + "][" + x + "]: " + node.outputValues[i][z][y][x]);
                        if (deltaMods == null) {
                            outputSum += node.outputValues[i][z][y][x];
                        } else {
                            outputSum += node.outputValues[i][z][y][x] * deltaMods[i][z][y][x];
                        }
                    }
                }
            }
        }

        return outputSum;
    }

    public static double[] getConvolutionalNodeNumericGradient(ConvolutionalNode node, double[][][][] inputs, double[] weights, int batchSize, int sizeZ, int sizeY, int sizeX) {
        double[] numericGradient = new double[weights.length];
        double[] testWeights = new double[weights.length];

        double H = 0.0000001;
        for (int i = 0; i < numericGradient.length; i++) {
            System.arraycopy(weights, 0, testWeights, 0, weights.length);

            testWeights[i] = weights[i] + H;
            double error1 = getConvolutionalNodeOutput(node, inputs, testWeights, batchSize, sizeZ, sizeY, sizeX, null);

            testWeights[i] = weights[i] - H;
            double error2 = getConvolutionalNodeOutput(node, inputs, testWeights, batchSize, sizeZ, sizeY, sizeX, null);

            numericGradient[i] = (error1 - error2) / (2.0 * H);

            Log.trace("numericGradient[" + i + "]: " + numericGradient[i] + ", error1: " + error1 + ", error2: " + error2 + ", testWeight1: " + (weights[i] + H) + ", testWeight2: "     + (weights[i] - H));
        }

        return numericGradient;
    }

    public static void testConvolutionalEdge(int seed, int batchSize, int inputPadding, int inputSizeZ, int inputSizeY, int inputSizeX, int outputPadding, int outputSizeZ, int outputSizeY, int outputSizeX, boolean checkGradients) throws NeuralNetworkException {
        //activation functions won't be used on the nodes so it doesn't matter which one we use
        int inputLayer = 1;
        int inputNumber = 1;
        ConvolutionalNode inputNode = new ConvolutionalNode(inputLayer, inputNumber, NodeType.HIDDEN, ActivationType.RELU, inputPadding, batchSize, inputSizeZ, inputSizeY, inputSizeX, false, 0.0, false, 0.0);

        int outputLayer = 2;
        int outputNumber = 1;
        ConvolutionalNode outputNode = new ConvolutionalNode(outputLayer, outputNumber, NodeType.HIDDEN, ActivationType.RELU, outputPadding, batchSize, outputSizeZ, outputSizeY, outputSizeX, false, 0.0, false, 0.0);

        int edgeSizeZ = inputSizeZ - outputSizeZ + 1;
        int edgeSizeY = (inputSizeY + (2 * inputPadding)) - outputSizeY + 1;
        int edgeSizeX = (inputSizeX + (2 * inputPadding)) - outputSizeX + 1;
        Log.info("Creating input convolutional node with size (" + inputSizeZ + "x" + inputSizeY + "x" + inputSizeX + ")");
        Log.info("Creating output convolutional node with size (" + outputSizeZ + "x" + outputSizeY + "x" + outputSizeX + ")");
        Log.info("Creating convolutional edge with size (" + edgeSizeZ + "x" + edgeSizeY + "x" + edgeSizeX + ")");

        ConvolutionalEdge edge = new ConvolutionalEdge(inputNode, outputNode, edgeSizeZ, edgeSizeY, edgeSizeX);


        Random generator = new Random(seed);

        //save the input values so we can use them to calculate the numeric gradient for the CNN node
        double[][][][] inputValues = new double[batchSize][inputSizeZ][inputSizeY + (2 * inputPadding)][inputSizeX + (2 * inputPadding)];
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < inputSizeZ; z++) {
                for (int y = 0; y < inputSizeY + (2 * inputPadding); y++) {
                    for (int x = 0; x < inputSizeX + (2 * inputPadding); x++) {
                        inputValues[i][z][y][x] = (generator.nextDouble() * 11.0) - 3.0;

                        inputNode.outputValues[i][z][y][x] = inputValues[i][z][y][x];
                    }
                }
            }
        }

        double[] weights = new double[edgeSizeZ * edgeSizeY * edgeSizeX];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (generator.nextDouble() * 0.10) - 0.05;
        }

        edge.setWeights(0, weights);

        edge.propagateForward(inputNode.outputValues);

        //do not uncomment these as they will overwrite the correct values I've generated for the test
        String extraName = "_convolutional_edge_" + inputPadding + "_" + inputSizeZ + "_" + inputSizeY + "_" + inputSizeX + "_" + outputPadding + "_" + outputSizeZ + "_" + outputSizeY + "_" + outputSizeX;
        String extraText = "seed: " + seed + ", batchSize: " + batchSize + ", inputPadding: " + inputPadding + ", inputSizeZ: " + inputSizeZ + ", inputSizeY: " + inputSizeY + ", inputSizeX: " + inputSizeX + ", outputPadding: " + outputPadding + ", outputSizeZ: " + outputSizeZ + ", outputSizeY: " + outputSizeY + ", outputSizeX: " + outputSizeX;


        if (generatingTestValues) TestValues.writeArray4d(inputNode.outputValues, "inputValues" + extraName, seed);
        if (generatingTestValues) TestValues.writeArray4d(outputNode.inputValues, "outputValues" + extraName, seed);

        Log.info("Checking inputValues for " + extraText);
        TestValues.testArray4d(inputNode.outputValues, TestValues.readArray4d("inputValues" + extraName, seed), "inputValues" + extraName, seed);

        Log.info("Checking outputValues for " + extraText);
        TestValues.testArray4d(outputNode.inputValues, TestValues.readArray4d("outputValues" + extraName, seed), "outputValues" + extraName, seed);

        if (!checkGradients) return; //don't test the gradients for PA3-1

        double[] numericGradient = getConvolutionalEdgeNumericGradient(edge, inputValues, weights);
        Log.info("numeric gradient: " + Arrays.toString(numericGradient));

        //don't uncomment this as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray1d(numericGradient, "numeric_gradient" + extraName, seed);

        Log.info("Checking numeric_gradient for " + extraText);
        TestValues.testArray1d(numericGradient, TestValues.readArray1d("numeric_gradient" + extraName, seed), "numeric_gradient" + extraName, seed);

        //getConvolutionalEdgeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX);

        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < outputSizeZ; z++) {
                for (int y = 0; y < outputSizeY; y++) {
                    for (int x = 0; x < outputSizeX; x++) {
                        outputNode.delta[i][z][y][x] = 1;
                    }
                }
            }
        }

        edge.propagateBackward(outputNode.delta);

        double[] deltas = new double[edgeSizeZ * edgeSizeY * edgeSizeX];
        edge.getDeltas(0, deltas);

        for (int j = 0; j < deltas.length; j++) {
            Log.debug("node.deltas[" + j + "]: " + deltas[j]);
        }

        Log.info("checking to see if numeric gradient and backprop deltas are close enough for convolutional edge.");
        if (!BasicTests.gradientsCloseEnough(numericGradient, deltas)) {
            throw new NeuralNetworkException("backprop vs numeric gradient check failed for " + extraText);
        }

    }

    public static double getConvolutionalEdgeOutput(ConvolutionalEdge edge, double[][][][] inputs, double[] weights) {
        edge.reset();
        edge.inputNode.reset();
        edge.outputNode.reset();
        edge.setWeights(0, weights);

        for (int i = 0; i < edge.inputNode.batchSize; i++) {
            for (int z = 0; z < edge.inputNode.sizeZ; z++) {
                for (int y = 0; y < edge.inputNode.sizeY; y++) {
                    for (int x = 0; x < edge.inputNode.sizeX; x++) {
                        edge.inputNode.outputValues[i][z][y][x] = inputs[i][z][y][x];
                    }
                }
            }
        }

        edge.propagateForward(edge.inputNode.outputValues);

        double outputSum = 0.0;
        for (int i = 0; i < edge.outputNode.batchSize; i++) {
            for (int z = 0; z < edge.outputNode.sizeZ; z++) {
                for (int y = 0; y < edge.outputNode.sizeY; y++) {
                    for (int x = 0; x < edge.outputNode.sizeX; x++) {
                        outputSum += edge.outputNode.inputValues[i][z][y][x];
                    }
                }
            }
        }

        return outputSum;
    }

    public static double[] getConvolutionalEdgeNumericGradient(ConvolutionalEdge edge, double[][][][] inputs, double[] weights) {
        double[] numericGradient = new double[weights.length];
        double[] testWeights = new double[weights.length];

        double H = 0.0000001;
        for (int i = 0; i < numericGradient.length; i++) {
            System.arraycopy(weights, 0, testWeights, 0, weights.length);

            testWeights[i] = weights[i] + H;
            double error1 = getConvolutionalEdgeOutput(edge, inputs, testWeights);

            testWeights[i] = weights[i] - H;
            double error2 = getConvolutionalEdgeOutput(edge, inputs, testWeights);

            numericGradient[i] = (error1 - error2) / (2.0 * H);

            Log.trace("numericGradient[" + i + "]: " + numericGradient[i] + ", error1: " + error1 + ", error2: " + error2 + ", testWeight1: " + (weights[i] + H) + ", testWeight2: "     + (weights[i] - H));
        }

        return numericGradient;
    }



    public static void testPoolingEdge(int seed, int batchSize, int inputPadding, int inputSizeZ, int inputSizeY, int inputSizeX, int outputPadding, int outputSizeZ, int outputSizeY, int outputSizeX, int poolSize, int stride, boolean checkGradients) throws NeuralNetworkException {
        //activation functions won't be used on the nodes so it doesn't matter which one we use
        int inputLayer = 1;
        int inputNumber = 1;
        ConvolutionalNode inputNode = new ConvolutionalNode(inputLayer, inputNumber, NodeType.HIDDEN, ActivationType.RELU, inputPadding, batchSize, inputSizeZ, inputSizeY, inputSizeX, false, 0.0, false, 0.0);

        int outputLayer = 2;
        int outputNumber = 1;
        ConvolutionalNode outputNode = new ConvolutionalNode(outputLayer, outputNumber, NodeType.HIDDEN, ActivationType.RELU, outputPadding, batchSize, outputSizeZ, outputSizeY, outputSizeX, false, 0.0, false, 0.0);

        int edgeSizeZ = inputSizeZ;
        int edgeSizeY = inputSizeY;
        int edgeSizeX = inputSizeX;
        Log.info("Creating input pooling node with size (" + inputSizeZ + "x" + inputSizeY + "x" + inputSizeX + ")");
        Log.info("Creating output pooling node with size (" + outputSizeZ + "x" + outputSizeY + "x" + outputSizeX + ")");
        Log.info("Creating pooling edge with pool size: " + poolSize + ", stride: " + stride);

        PoolingEdge edge = new PoolingEdge(inputNode, outputNode, poolSize, stride);

        Random generator = new Random(seed);

        //save the input values so we can use them to calculate the numeric gradient for the CNN node
        double[][][][] inputValues = new double[batchSize][inputSizeZ][inputSizeY][inputSizeX];
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < inputSizeZ; z++) {
                for (int y = 0; y < inputSizeY; y++) {
                    for (int x = 0; x < inputSizeX; x++) {
                        inputValues[i][z][y][x] = (generator.nextDouble() * 11.0) - 3.0;

                        inputNode.outputValues[i][z][y][x] = inputValues[i][z][y][x];
                    }
                }
            }
        }

        //you can uncomment this to see what the input of the pooling operation is
        //TestValues.printArray4d("inputNode.outputValues", inputNode.outputValues);

        double[] weights = new double[0];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (generator.nextDouble() * 0.10) - 0.05;
        }

        edge.setWeights(0, weights);

        edge.propagateForward(inputNode.outputValues);
        //you can uncomment this to see what the output of the pooling operation is
        //TestValues.printArray4d("outputNode.inputValues", outputNode.inputValues);

        //do not uncomment these as they will overwrite the correct values I've generated for the test
        String extraName = "_pooling_edge_" + inputPadding + "_" + inputSizeZ + "_" + inputSizeY + "_" + inputSizeX + "_" + outputPadding + "_" + outputSizeZ + "_" + outputSizeY + "_" + outputSizeX;
        String extraText = "seed: " + seed + ", batchSize: " + batchSize + ", inputPadding: " + inputPadding + ", inputSizeZ: " + inputSizeZ + ", inputSizeY: " + inputSizeY + ", inputSizeX: " + inputSizeX + ", outputPadding: " + outputPadding + ", outputSizeZ: " + outputSizeZ + ", outputSizeY: " + outputSizeY + ", outputSizeX: " + outputSizeX;

        if (generatingTestValues) TestValues.writeArray4d(inputNode.outputValues, "inputValues" + extraName, seed);
        if (generatingTestValues) TestValues.writeArray4d(outputNode.inputValues, "outputValues" + extraName, seed);

        Log.info("Checking inputValues for " + extraText);
        TestValues.testArray4d(inputNode.outputValues, TestValues.readArray4d("inputValues" + extraName, seed), "inputValues" + extraName, seed);

        Log.info("Checking outputValues for " + extraText);
        TestValues.testArray4d(outputNode.inputValues, TestValues.readArray4d("outputValues" + extraName, seed), "outputValues" + extraName, seed);

        if (!checkGradients) return; //don't test the gradients for PA3-1


        //You can uncomment these to see what the pooling operation has done
        //TestValues.printArray4d("edge.poolDelta", edge.poolDelta);
        //TestValues.printArray4d("inputNode.outputValues", inputNode.outputValues);
        //TestValues.printArray4d("outputNode.inputValues", outputNode.inputValues);


        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < outputSizeZ; z++) {
                for (int y = 0; y < outputSizeY; y++) {
                    for (int x = 0; x < outputSizeX; x++) {
                        outputNode.delta[i][z][y][x] = 1;
                    }
                }
            }
        }

        edge.propagateBackward(outputNode.delta);

        if (generatingTestValues) TestValues.writeArray4d(inputNode.delta, "inputDelta" + extraName, seed);

        Log.info("Checking inputDelta for " + extraText);
        TestValues.testArray4d(inputNode.delta, TestValues.readArray4d("inputDelta" + extraName, seed), "inputDelta" + extraName, seed);

        //you can uncomment this to see what the resulting deltas are
        //TestValues.printArray4d("inputNode.delta", inputNode.delta);


        //save the delta from backprop so we can compare to the numeric gradient
        double[] backpropDelta = new double[batchSize * inputSizeZ * inputSizeY * inputSizeX];
        double[] numericDelta = new double[batchSize * inputSizeZ * inputSizeY * inputSizeX];

        int current = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < inputSizeZ; z++) {
                for (int y = 0; y < inputSizeY; y++) {
                    for (int x = 0; x < inputSizeX; x++) {
                        backpropDelta[current] = inputNode.delta[i][z][y][x];
                        current++;
                    }
                }
            }
        }

        double H = 0.0000001;

        current = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < inputSizeZ; z++) {
                for (int y = 0; y < inputSizeY; y++) {
                    for (int x = 0; x < inputSizeX; x++) {
                        inputNode.reset();
                        outputNode.reset();
                        edge.reset();
                        double original = inputValues[i][z][y][x];
                        inputValues[i][z][y][x] = original + H;

                        //this will make sure the same nodes are dropped out each pass
                        setOutputs(inputNode, inputValues);
                        edge.propagateForward(inputNode.outputValues);
                        double error1 = getSum(outputNode.inputValues);
                        //TestValues.printArray4d("inputNode.outputValues", inputNode.outputValues);
                        //TestValues.printArray4d("outputNode.inputValues", outputNode.inputValues);
                        //TestValues.printArray4d("edge.poolDelta", edge.poolDelta);

                        inputValues[i][z][y][x] = original - H;

                        inputNode.reset();
                        outputNode.reset();
                        edge.reset();

                        //this will make sure the same nodes are dropped out each pass
                        setOutputs(inputNode, inputValues);
                        edge.propagateForward(inputNode.outputValues);
                        double error2 = getSum(outputNode.inputValues);
                        //TestValues.printArray4d("inputNode.outputValues", inputNode.outputValues);
                        //TestValues.printArray4d("outputNode.inputValues", outputNode.inputValues);
                        //TestValues.printArray4d("edge.poolDelta", edge.poolDelta);

                        inputValues[i][z][y][x] = original;

                        numericDelta[current] = (error1 - error2) / (2.0 * H);
                        current++;

                        //Log.info("numericDelta[" + (current - 1) + "]: " + numericDelta[current - 1] + ", backpropDelta[" + (current - 1) + "]: " + backpropDelta[current - 1]);
                        if (Math.abs(numericDelta[current - 1] - backpropDelta[current - 1]) > 1e-6) {
                            System.out.println("errors[" + i + "][" + z + "][" + y + "][" + x + "]: first: " + error1 + ", second: " + error2 + ", difference: " + (error1 - error2));
                            System.err.println("Error in calculating deltas!");
                            Log.info("numericDelta[" + (current - 1) + "]: " + numericDelta[current - 1] + ", backpropDelta[" + (current - 1) + "]: " + backpropDelta[current - 1]);
                            System.exit(1);
                        }
                    }
                }
            }
        }
    }

    public static void setOutputs(ConvolutionalNode node, double[][][][] m) {
        for (int i = 0; i < m.length; i++) {
            for (int z = 0; z < m[i].length; z++) {
                for (int y = 0; y < m[i][z].length; y++) {
                    for (int x = 0; x < m[i][z][y].length; x++) {
                        node.outputValues[i][z][y][x] = m[i][z][y][x];
                    }
                }
            }
        }
    }

    public static double getSum(double[][][][] m) {
        double sum = 0.0;
        for (int i = 0; i < m.length; i++) {
            for (int z = 0; z < m[i].length; z++) {
                for (int y = 0; y < m[i][z].length; y++) {
                    for (int x = 0; x < m[i][z][y].length; x++) {
                        sum += m[i][z][y][x];
                    }
                }
            }
        }
        return sum;
    }
}

