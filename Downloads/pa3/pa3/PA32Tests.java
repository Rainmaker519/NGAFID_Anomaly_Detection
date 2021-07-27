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



public class PA32Tests {
    public static final boolean checkGradients = true;

    public static void main(String[] arguments) {
        //test convolutional nodes
        try {
            //test a convolutional node with to make sure
            //a basic forward and backward pass is working
            //first with a batch size of 1 and 1 channel
            int batchSize = 1;
            int sizeZ = 1;
            int sizeY = 10;
            int sizeX = 10;
            PA31Tests.testConvolutionalNode(10 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(32 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(43 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);

            //then with a batch size of 10 and 1 channel
            batchSize = 10;
            PA31Tests.testConvolutionalNode(110 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(332 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(443 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);

            //then with a batch size of 1 and 3 channels
            batchSize = 1;
            sizeZ = 3;
            PA31Tests.testConvolutionalNode(112 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(334 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(445 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);

            //then with a batch size of 10 and 3 channels
            batchSize = 10;
            sizeZ = 3;
            PA31Tests.testConvolutionalNode(11220 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(33442 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
            PA31Tests.testConvolutionalNode(44553 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, false, 0.0, false, 0.0);
        } catch (NeuralNetworkException e) {
            System.err.println("ConvolutionalNode test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }

        //test convolutional edges
        try {
            int batchSize = 1;
            int inputPadding = 0;
            int inputSizeZ = 1;
            int inputSizeY = 10;
            int inputSizeX = 10;
            int outputPadding = 0;
            int outputSizeZ = 1;
            int outputSizeY = 5;
            int outputSizeX = 5;

            PA31Tests.testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            PA31Tests.testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            PA31Tests.testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            PA31Tests.testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

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

            PA31Tests.testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            PA31Tests.testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            PA31Tests.testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            PA31Tests.testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

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

            PA31Tests.testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            PA31Tests.testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            PA31Tests.testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            PA31Tests.testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

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

            PA31Tests.testConvolutionalEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            PA31Tests.testConvolutionalEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(221 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(332 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 1;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(1100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            batchSize = 10;
            inputSizeZ = 3;
            outputSizeZ = 3;
            PA31Tests.testConvolutionalEdge(10100 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(21211 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(32322 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 2;
            PA31Tests.testConvolutionalEdge(101010 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(212121 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(323232 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

            inputSizeZ = 3;
            outputSizeZ = 1;
            PA31Tests.testConvolutionalEdge(1010101 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(2121212 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);
            PA31Tests.testConvolutionalEdge(3232323 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, checkGradients);

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

            PA31Tests.testPoolingEdge(11 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(22 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(33 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);

            inputSizeY = 6;
            inputSizeX = 6;
            outputSizeY = 2;
            outputSizeX = 2;
            poolSize = 3;
            stride = 3;
            PA31Tests.testPoolingEdge(110 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(220 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(330 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);

            inputSizeY = 5;
            inputSizeX = 5;
            outputSizeY = 2;
            outputSizeX = 2;
            poolSize = 3;
            stride = 2;
            PA31Tests.testPoolingEdge(1103 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(2204 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(3305 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);

            inputSizeY = 11;
            inputSizeX = 11;
            outputSizeY = 5;
            outputSizeX = 5;
            poolSize = 3;
            stride = 2;
            PA31Tests.testPoolingEdge(1103 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(2204 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
            PA31Tests.testPoolingEdge(3305 /*seed*/, batchSize, inputPadding, inputSizeZ, inputSizeY, inputSizeX, outputPadding, outputSizeZ, outputSizeY, outputSizeX, poolSize, stride, checkGradients);
        } catch (NeuralNetworkException e) {
            System.err.println("PoolingEdge test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}

