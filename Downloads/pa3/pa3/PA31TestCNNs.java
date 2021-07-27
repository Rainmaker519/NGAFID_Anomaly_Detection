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



public class PA31TestCNNs {
    public static final int NUMBER_REPEATS = 1;
    public static final boolean generatingTestValues = false; //do not modify or it will overwrite my correctly generated test values
    public static final boolean checkGradients = false;

    public static void main(String[] arguments) {
        //test LeNet-5
        try {
            //test for MNIST
            MNISTDataSet mnistTrain = new MNISTDataSet("./datasets/train-images-idx3-ubyte", "./datasets/train-labels-idx1-ubyte", 60000);
            mnistTrain.resize(1000);

            boolean useDropout = false;
            double inputDropoutRate = 0.0;
            double hiddenDropoutRate = 0.0;
            boolean useBatchNormalization = false;
            double alpha = 0.0;

            testCNN(mnistTrain, "mnist_small_no_pool", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(mnistTrain, "mnist_small", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(mnistTrain, "mnist_lenet5", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

            testCNN(mnistTrain, "mnist_small_no_pool", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(mnistTrain, "mnist_small", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(mnistTrain, "mnist_lenet5", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

            CIFARDataSet cifarTrain = new CIFARDataSet(new String[]{"./datasets/cifar-10-batches-bin/data_batch_1.bin", "./datasets/cifar-10-batches-bin/data_batch_2.bin", "./datasets/cifar-10-batches-bin/data_batch_3.bin", "./datasets/cifar-10-batches-bin/data_batch_4.bin", "./datasets/cifar-10-batches-bin/data_batch_5.bin"}, 10000);
            cifarTrain.resize(1000);

            testCNN(cifarTrain, "cifar_small_no_pool", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(cifarTrain, "cifar_small", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(cifarTrain, "cifar_lenet5", 1, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

            testCNN(cifarTrain, "cifar_small_no_pool", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(cifarTrain, "cifar_small", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);
            testCNN(cifarTrain, "cifar_lenet5", 5, checkGradients, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);


        } catch (NeuralNetworkException e) {
            System.err.println("CNN test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        } 
    }

    public static void testCNN(ImageDataSet imageData, String networkType, int batchSize, boolean checkGradients, boolean useDropout, double inputDropoutRate, double hiddenDropoutRate, boolean useBatchNormalization, double alpha) throws NeuralNetworkException {
        ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(LossFunction.SOFTMAX, useDropout, inputDropoutRate, hiddenDropoutRate, useBatchNormalization, alpha);

        if (networkType.equals("mnist_small_no_pool")) {
            cnn.createSmallNoPool(ActivationType.LEAKY_RELU5, batchSize, imageData.getNumberChannels(), imageData.getNumberRows(), imageData.getNumberCols(), 2, imageData.getNumberClasses());
        } else if (networkType.equals("mnist_small")) {
            cnn.createSmall(ActivationType.LEAKY_RELU5, batchSize, imageData.getNumberChannels(), imageData.getNumberRows(), imageData.getNumberCols(), 2, imageData.getNumberClasses());
        } else if (networkType.equals("mnist_lenet5")) {
            cnn.createLeNet5(ActivationType.LEAKY_RELU5, batchSize, imageData.getNumberChannels(), imageData.getNumberRows(), imageData.getNumberCols(), 2, imageData.getNumberClasses());
        } else if (networkType.equals("cifar_small_no_pool")) {
            cnn.createSmallNoPool(ActivationType.LEAKY_RELU5, batchSize, imageData.getNumberChannels(), imageData.getNumberRows(), imageData.getNumberCols(), 0, imageData.getNumberClasses());
        } else if (networkType.equals("cifar_small")) {
            cnn.createSmall(ActivationType.LEAKY_RELU5, batchSize, imageData.getNumberChannels(), imageData.getNumberRows(), imageData.getNumberCols(), 0, imageData.getNumberClasses());
        } else if (networkType.equals("cifar_lenet5")) {
            cnn.createLeNet5(ActivationType.LEAKY_RELU5, batchSize, imageData.getNumberChannels(), imageData.getNumberRows(), imageData.getNumberCols(), 0, imageData.getNumberClasses());
        } 


        int numberWeights = cnn.getNumberWeights();
        Log.info("numberWeights: " + numberWeights);

        if (useBatchNormalization) {
            if (networkType.equals("mnist_small_no_pool")) {
                if (numberWeights != 8758) {
                    throw new NeuralNetworkException(networkType + " (batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 8758");
                }
            } else if (networkType.equals("mnist_small")) {
                if (numberWeights != 6822) {
                    throw new NeuralNetworkException(networkType + " (batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 6822");
                }
            } else if (networkType.equals("mnist_lenet5")) {
                if (numberWeights != 69150) {
                    throw new NeuralNetworkException(networkType + " (batch norm) Number of weights was incorrect (" + numberWeights + "), should have been 69150");
                }
            } else if (networkType.equals("cifar_small_no_pool")) {
                if (numberWeights != 10110) {
                    throw new NeuralNetworkException(networkType + " (batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 10110");
                }
            } else if (networkType.equals("cifar_small")) {
                if (numberWeights != 8174) {
                    throw new NeuralNetworkException(networkType + " (batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 8174");
                }
            } else if (networkType.equals("cifar_lenet5")) {
                if (numberWeights != 69450) {
                    throw new NeuralNetworkException(networkType + " (batch norm) Number of weights was incorrect (" + numberWeights + "), should have been 69450");
                }
            } 

        } else {
            if (networkType.equals("mnist_small_no_pool")) {
                if (numberWeights != 8722) {
                    throw new NeuralNetworkException(networkType + " (no batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 8732");
                }
            } else if (networkType.equals("mnist_small")) {
                if (numberWeights != 6786) {
                    throw new NeuralNetworkException(networkType + " (no batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 6796");
                }
            } else if (networkType.equals("mnist_lenet5")) {
                if (numberWeights != 68654) {
                    throw new NeuralNetworkException(networkType + " (no batch norm) Number of weights was incorrect (" + numberWeights + "), should have been 68654");
                }
            } else if (networkType.equals("cifar_small_no_pool")) {
                if (numberWeights != 10074) {
                    throw new NeuralNetworkException(networkType + " (no batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 10084");
                }
            } else if (networkType.equals("cifar_small")) {
                if (numberWeights != 8138) {
                    throw new NeuralNetworkException(networkType + " (no batch norm) CNN Number of weights was incorrect (" + numberWeights + "), should have been 8138");
                }
            } else if (networkType.equals("cifar_lenet5")) {
                if (numberWeights != 68954) {
                    throw new NeuralNetworkException(networkType + " (no batch norm) Number of weights was incorrect (" + numberWeights + "), should have been 68954");
                }
            } 
        }


        int seed = 1337;
        Random generator = new Random(seed);
        double[] weights = new double[numberWeights];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (generator.nextDouble() * 0.10) - 0.05;
        }

        cnn.setWeights(weights);

        double loss = cnn.forwardPass(imageData, 0, batchSize, true);
        double[][] outputValues = cnn.getOutputValues(batchSize);


        String extraName = "_" + networkType + "_" + batchSize + "_" + useDropout + "_" + inputDropoutRate + "_" + hiddenDropoutRate + "_" + useBatchNormalization + "_" + alpha; 
        String extraText = "network: " + networkType + ", seed: " + seed + ", batchSize: " + batchSize + ", dropout: " + useDropout + ", input dropout: " + inputDropoutRate + ", hidden dropout:" + hiddenDropoutRate + ", batch norm:" + useBatchNormalization + ", alpha: " + alpha;
        if (generatingTestValues) TestValues.writeValue(loss, "loss" + extraName, seed);
        Log.info("Checking loss for " + extraText);
        TestValues.testValue(loss, TestValues.readValue("loss" + extraName, seed), "loss" + extraName, seed);

        if (generatingTestValues) TestValues.writeArray2d(outputValues, "outputValues" + extraName, seed);
        Log.info("Checking outputValues for " + extraText);
        TestValues.testArray2d(outputValues, TestValues.readArray2d("outputValues" + extraName, seed), "outputValues" + extraName, seed);

        if (!checkGradients) return; //don't check the gradients for PA3-1

        double[] backpropGradient = cnn.getGradient(imageData, 0, batchSize);
        Log.info("Got backprop gradient!");

        double[] numericGradient = cnn.getNumericGradient(imageData, 0, batchSize);
        //Log.info("numeric gradient: " + Arrays.toString(numericGradient));

        //don't uncomment this as it will overwrite my precomputed correct values
        if (generatingTestValues) TestValues.writeArray1d(numericGradient, "numeric_gradient" + extraName, seed);

        Log.info("Checking numeric_gradient for " + extraText);
        cnn.printGradients(numericGradient, backpropGradient);

        TestValues.testArray1d(numericGradient, TestValues.readArray1d("numeric_gradient" + extraName, seed), "numeric_gradient" + extraName, seed);


        Log.info("Checking backprop vs numeric gradients for network: " + networkType);
        if (!BasicTests.gradientsCloseEnough(numericGradient, backpropGradient)) {
            cnn.printGradients(numericGradient, backpropGradient);

            throw new NeuralNetworkException("backprop vs numeric gradient check failed from and images 0 to " + batchSize + " for " + extraText);
        }

    }

}

