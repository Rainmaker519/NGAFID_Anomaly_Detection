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



public class PA33Tests {
    public static final boolean checkGradients = true;

    public static void main(String[] arguments) {
        //test convolutional nodes
        try {
            //test a convolutional node first with
            //dropout to make sure it's working
            boolean useDropout = true;
            double dropoutRate = 0.3;
            boolean useBatchNormalization = false;
            double alpha = 0.9;

            //first with a batch size of 1 and 1 channel

            int batchSize = 1;
            int sizeZ = 1;
            int sizeY = 5;
            int sizeX = 5;

            testRegularization(10 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(32 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(43 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 3;

            testRegularization(110 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(332 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(443 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 1;
            sizeZ = 3;
            testRegularization(112 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(334 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(445 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 3;
            testRegularization(11220 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(33442 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(44553 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            //test a convolutional node with
            //batch normalization but not dropout 
            //to make sure batch norm is working
            useDropout = false;
            dropoutRate = 0.3;
            useBatchNormalization = true;
            alpha = 0.1;

            batchSize = 1;
            sizeZ = 1;

            testRegularization(10 /*seed*/, ActivationType.NONE, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(13 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(32 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(43 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 3;

            testRegularization(110 /*seed*/, ActivationType.NONE, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(113 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(332 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(443 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 1;
            sizeZ = 3;
            testRegularization(112 /*seed*/, ActivationType.NONE, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(1123 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(334 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(445 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 3;
            testRegularization(11220 /*seed*/, ActivationType.NONE, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(11223 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(33442 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(44553 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            //test a convolutional node with both
            //batch normalization and dropout 
            //to make sure they are working together
            useDropout = true;
            dropoutRate = 0.3;
            useBatchNormalization = true;
            alpha = 0.1;

            batchSize = 1;
            sizeZ = 1;

            testRegularization(10 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(32 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(43 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 3;

            testRegularization(110 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(332 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(443 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 1;
            sizeZ = 3;
            testRegularization(112 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(334 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(445 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

            batchSize = 3;
            testRegularization(11220 /*seed*/, ActivationType.RELU, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(33442 /*seed*/, ActivationType.RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);
            testRegularization(44553 /*seed*/, ActivationType.LEAKY_RELU5, batchSize, sizeZ, sizeY, sizeX, checkGradients, useDropout, dropoutRate, useBatchNormalization, alpha);

        } catch (NeuralNetworkException e) {
            System.err.println("ConvolutionalNode test failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void testRegularization(int seed, ActivationType activationType, int batchSize, int sizeZ, int sizeY, int sizeX, boolean checkGradients, boolean useDropout, double dropoutRate, boolean useBatchNormalization, double alpha) throws NeuralNetworkException {
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

        //we need to make sure we generate the same random values for dropout each
        //time so we can correctly calculate the numeric gradient
        node.generator = new Random(seed);
        node.propagateForward(true);

        //do not uncomment these as they will overwrite the correct values I've generated for the test
        String extraName = "_convolutional_node_" + activationType + "_" + batchSize + "_" + sizeZ + "_" + sizeY + "_" + sizeX + "_" + useDropout + "_" + dropoutRate + "_" + useBatchNormalization  + "_" + alpha;
        String extraText = "seed: " + seed + ", activationType: " + activationType + ", batchSize: " + batchSize + ", sizeZ: " + sizeZ + ", sizeY: " + sizeY + ", sizeX: " + sizeX + ", dropout: " + useDropout + ", dropoutRate: " + dropoutRate + ", batchNorm: " + useBatchNormalization + ", alpha: " + alpha;

        if (PA31Tests.generatingTestValues) TestValues.writeArray4d(node.inputValues, "inputValues" + extraName, seed);
        if (PA31Tests.generatingTestValues) TestValues.writeArray4d(node.outputValues, "outputValues" + extraName, seed);
        if (useDropout && PA31Tests.generatingTestValues) TestValues.writeArray4d(node.dropoutDelta, "dropoutDelta" + extraName, seed);

        Log.info("Checking inputValues for " + extraText);
        TestValues.testArray4d(node.inputValues, TestValues.readArray4d("inputValues" + extraName, seed), "inputValues" + extraName, seed);

        Log.info("Checking outputValues for " + extraText);
        TestValues.testArray4d(node.outputValues, TestValues.readArray4d("outputValues" + extraName, seed), "outputValues" + extraName, seed);

        if (useDropout) Log.info("Checking dropoutDelta for " + extraText);
        if (useDropout) TestValues.testArray4d(node.dropoutDelta, TestValues.readArray4d("dropoutDelta" + extraName, seed), "dropoutDelta" + extraName, seed);

        //you can uncomment these if you want to see the input, output values and the delta for dropout
        //TestValues.printArray4d("node.inputValues", node.inputValues);
        //TestValues.printArray4d("node.outputValues", node.outputValues);
        //if (useDropout) TestValues.printArray4d("node.dropoutDelta", node.dropoutDelta);

        double[][][][] deltaMods = new double[batchSize][sizeZ][sizeY][sizeX];

        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        deltaMods[i][z][y][x] = (generator.nextDouble() * 2.0) - 2.0;
                        node.delta[i][z][y][x] = deltaMods[i][z][y][x];
                    }
                }
            }
        }
        node.propagateBackward();

        //you can uncomment this if you want to see the delta caluclated by backprop
        //TestValues.printArray4d("node.delta", node.delta);

        if (PA31Tests.generatingTestValues) TestValues.writeArray4d(node.delta, "delta" + extraName, seed);

        Log.info("Checking delta for " + extraText);
        TestValues.testArray4d(node.delta, TestValues.readArray4d("delta" + extraName, seed), "delta" + extraName, seed);

        //save the delta from backprop so we can compare to the numeric gradient
        double[] backpropDelta = null;
        if (useBatchNormalization) {
            backpropDelta = new double[(batchSize * sizeZ * sizeY * sizeX) + 2];
        } else {
            backpropDelta = new double[batchSize * sizeZ * sizeY * sizeX];
        }

        int current = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        backpropDelta[current] = node.delta[i][z][y][x];
                        current++;
                    }
                }
            }
        }

        if (useBatchNormalization) {
            backpropDelta[current] = node.betaDelta;
            current++;
            backpropDelta[current] = node.gammaDelta;
        }


        double H = 0.0000001;

        double[] numericDelta = null;
        if (useBatchNormalization) {
            numericDelta = new double[(batchSize * sizeZ * sizeY * sizeX) + 2];
        } else {
            numericDelta = new double[batchSize * sizeZ * sizeY * sizeX];
        }

        current = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int z = 0; z < sizeZ; z++) {
                for (int y = 0; y < sizeY; y++) {
                    for (int x = 0; x < sizeX; x++) {
                        double original = inputValues[i][z][y][x];
                        inputValues[i][z][y][x] = original + H;

                        //this will make sure the same nodes are dropped out each pass
                        node.generator = new Random(seed);
                        //node.propagateForward(true);
                        double error1 = PA31Tests.getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX, deltaMods);
                        //TestValues.printArray4d("node.inputValues", node.inputValues);
                        //TestValues.printArray4d("node.outputValues", node.outputValues);
                        double output1 = node.outputValues[i][z][y][x];

                        inputValues[i][z][y][x] = original - H;

                        //this will make sure the same nodes are dropped out each pass
                        node.generator = new Random(seed);
                        //node.propagateForward(true);
                        double error2 = PA31Tests.getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX, deltaMods);
                        //TestValues.printArray4d("node.inputValues", node.inputValues);
                        //TestValues.printArray4d("node.outputValues", node.outputValues);
                        double output2 = node.outputValues[i][z][y][x];

                        inputValues[i][z][y][x] = original;

                        numericDelta[current] = (error1 - error2) / (2.0 * H);
                        current++;

                        //Log.info("numericDelta[" + (current - 1) + "]: " + numericDelta[current - 1] + ", backpropDelta[" + (current - 1) + "]: " + backpropDelta[current - 1]);
                    }
                }
            }
        }

        if (useBatchNormalization) {
            //calculate the numeric gradients for beta and gamma
            int betaWeightIndex = weights.length - 2;

            double original = weights[betaWeightIndex];
            weights[betaWeightIndex] = original + H;

            //this will make sure the same nodes are dropped out each pass
            node.generator = new Random(seed);
            //node.propagateForward(true);
            double error1 = PA31Tests.getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX, deltaMods);
            //TestValues.printArray4d("node.inputValues", node.inputValues);
            //TestValues.printArray4d("node.outputValues", node.outputValues);

            weights[betaWeightIndex] = original - H;

            //this will make sure the same nodes are dropped out each pass
            node.generator = new Random(seed);
            //node.propagateForward(true);
            double error2 = PA31Tests.getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX, deltaMods);
            //TestValues.printArray4d("node.inputValues", node.inputValues);
            //TestValues.printArray4d("node.outputValues", node.outputValues);

            weights[betaWeightIndex] = original;

            numericDelta[current] = (error1 - error2) / (2.0 * H);
            current++;

            int gammaWeightIndex = weights.length - 1;

            original = weights[gammaWeightIndex];
            weights[gammaWeightIndex] = original + H;

            //this will make sure the same nodes are dropped out each pass
            node.generator = new Random(seed);
            //node.propagateForward(true);
            error1 = PA31Tests.getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX, deltaMods);
            //TestValues.printArray4d("node.inputValues", node.inputValues);
            //TestValues.printArray4d("node.outputValues", node.outputValues);

            weights[gammaWeightIndex] = original - H;

            //this will make sure the same nodes are dropped out each pass
            node.generator = new Random(seed);
            //node.propagateForward(true);
            error2 = PA31Tests.getConvolutionalNodeOutput(node, inputValues, weights, batchSize, sizeZ, sizeY, sizeX, deltaMods);
            //TestValues.printArray4d("node.inputValues", node.inputValues);
            //TestValues.printArray4d("node.outputValues", node.outputValues);

            weights[gammaWeightIndex] = original;

            numericDelta[current] = (error1 - error2) / (2.0 * H);
            current++;
        }


        boolean hadError = false;
        if (useBatchNormalization) {
            current = 0;
            for (int i = 0; i < batchSize; i++) {
                for (int z = 0; z < sizeZ; z++) {
                    for (int y = 0; y < sizeY; y++) {
                        for (int x = 0; x < sizeX; x++) {
                            if (Math.abs(numericDelta[current] - backpropDelta[current]) > 1e-6) {
                                System.out.println("Error in calcualting deltas for outputs[" + i + "][" + z + "][" + y + "][" + x + "]: numericDelta[" + (current) + "]: " + numericDelta[current] + ", backpropDelta[" + (current) + "]: " + backpropDelta[current] + ", difference: " + (numericDelta[current] - backpropDelta[current]));
                                hadError = true;
                            }
                            current++;
                        }
                    }
                }
            }

            if (Math.abs(numericDelta[current] - backpropDelta[current]) > 1e-6) {
                System.out.println("Error in calcualting deltas batchnorm beta: numericDelta[" + (current) + "]: " + numericDelta[current] + ", backpropDelta[" + (current) + "]: " + backpropDelta[current] + ", difference: " + (numericDelta[current] - backpropDelta[current]));
                hadError = true;
            }
            current++;

            if (Math.abs(numericDelta[current] - backpropDelta[current]) > 1e-6) {
                System.out.println("Error in calcualting deltas batchnorm gamma: numericDelta[" + (current) + "]: " + numericDelta[current] + ", backpropDelta[" + (current) + "]: " + backpropDelta[current] + ", difference: " + (numericDelta[current] - backpropDelta[current]));
                hadError = true;
            }
        }


        if (!BasicTests.gradientsCloseEnough(numericDelta, backpropDelta)) {
            node.printGradients(0, numericDelta, backpropDelta);

            throw new NeuralNetworkException("backprop vs numeric delta check failed on for ConvolutionalNode regularization tests, " + extraText);
        }
    }
}
