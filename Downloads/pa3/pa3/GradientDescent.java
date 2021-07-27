/**
 * This is a helpful class to let you test the required
 * functionality for Programming Assignment 2 - Part 3.
 *
 */
import java.util.Arrays;
import java.util.List;

import data.ImageDataSet;
import data.CIFARDataSet;
import data.MNISTDataSet;
import data.Image;

import network.ActivationType;
import network.LossFunction;
import network.ConvolutionalNeuralNetwork;
import network.NeuralNetworkException;
import network.NodeType;

import util.Log;
import util.Vector;


public class GradientDescent {
    public static void helpMessage() {
        Log.info("Usage:");
        Log.info("\tjava GradientDescent <data set> <network type> <initialization type> <batch size> <loss function> <epochs> <bias> <learning rate> <mu> <use dropout> <input dropout rate> <hidden dropout rate> <use batch normalization> <batch norm alpha>");
        Log.info("\t\tdata set can be: 'mnist' or 'cifar'");
        Log.info("\t\tnetwork type can be: 'small_no_pool', 'small' or 'lenet5'");
        Log.info("\t\tinitialization type can be: 'xavier' or 'kaiming'");
        Log.info("\t\tbatch size should be > 0, and should equally divide both the number of images in the training and in the testing data sets");
        Log.info("\t\tloss function can be: 'svm' or 'softmax'");
        Log.info("\t\tepochs is an integer > 0");
        Log.info("\t\tbias is a double");
        Log.info("\t\tlearning rate is a double usually small and > 0");
        Log.info("\t\tmu is a double < 1 and typical values are 0.5, 0.9, 0.95 and 0.99");
        Log.info("\t\tuse dropout is 0 (no dropout) or 1 (dropout)");
        Log.info("\t\tinput dropout rate is a double > 0, ignored if use dropout is 0");
        Log.info("\t\thidden dropout rate is a double > 0, ignored if use dropout is 0");
        Log.info("\t\tuse batch normalization is 0 (no batch normalization) or 1 (batch normalization)");
        Log.info("\t\tbatch norm alpha is usually 0.1 or 0.05");
    }

    public static void main(String[] arguments) {
        if (arguments.length != 14) {
            helpMessage();
            System.exit(1);
        }

        String dataSetName = arguments[0];
        String networkType = arguments[1];
        String initializationType = arguments[2];
        int batchSize = Integer.parseInt(arguments[3]);
        String lossFunctionName = arguments[4];
        int epochs = Integer.parseInt(arguments[5]);
        double bias = Double.parseDouble(arguments[6]);
        double learningRate = Double.parseDouble(arguments[7]);
        double mu = Double.parseDouble(arguments[8]);
        int useDropout = Integer.parseInt(arguments[9]);
        double inputDropoutRate = Double.parseDouble(arguments[10]);
        double hiddenDropoutRate = Double.parseDouble(arguments[11]);
        int useBatchNormalization = Integer.parseInt(arguments[12]);
        double alpha = Double.parseDouble(arguments[13]);

        Log.info("inputDropoutRate: " + inputDropoutRate + ", hiddenDropoutRate: " + hiddenDropoutRate);


        ImageDataSet trainingDataSet = null;
        ImageDataSet testingDataSet = null;

        if (dataSetName.equals("mnist")) {
            trainingDataSet = new MNISTDataSet("./datasets/train-images-idx3-ubyte", "./datasets/train-labels-idx1-ubyte", 60000);
            testingDataSet = new MNISTDataSet("./datasets/t10k-images-idx3-ubyte", "./datasets/t10k-labels-idx1-ubyte", 10000);
            networkType = "mnist_" + networkType;

        } else if (dataSetName.equals("cifar")) {
            trainingDataSet = new CIFARDataSet(new String[]{"./datasets/cifar-10-batches-bin/data_batch_1.bin", "./datasets/cifar-10-batches-bin/data_batch_2.bin", "./datasets/cifar-10-batches-bin/data_batch_3.bin", "./datasets/cifar-10-batches-bin/data_batch_4.bin", "./datasets/cifar-10-batches-bin/data_batch_5.bin"}, 10000);

            testingDataSet = new CIFARDataSet(new String[]{"./datasets/cifar-10-batches-bin/test_batch.bin"}, 10000);
            networkType = "cifar_" + networkType;

        } else {
            Log.fatal("unknown data set : " + dataSetName);
            System.exit(1);
        }

        //trainingDataSet.resize(10000);
        //testingDataSet.resize(10000);

        LossFunction lossFunction = LossFunction.NONE;
        if (lossFunctionName.equals("svm")) {
            Log.info("Using an SVM loss function.");
            lossFunction = LossFunction.SVM;
        } else if (lossFunctionName.equals("softmax")) {
            Log.info("Using an SOFTMAX loss function.");
            lossFunction = LossFunction.SOFTMAX;
        } else {
            Log.fatal("unknown loss function : " + lossFunctionName);
            System.exit(1);
        }

        ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(LossFunction.SOFTMAX, useDropout == 1, inputDropoutRate, hiddenDropoutRate, useBatchNormalization == 1, alpha);

        try {
            if (networkType.equals("mnist_small_no_pool")) {
                cnn.createSmallNoPool(ActivationType.LEAKY_RELU5, batchSize, trainingDataSet.getNumberChannels(), trainingDataSet.getNumberRows(), trainingDataSet.getNumberCols(), 2, trainingDataSet.getNumberClasses());
            } else if (networkType.equals("mnist_small")) {
                cnn.createSmall(ActivationType.LEAKY_RELU5, batchSize, trainingDataSet.getNumberChannels(), trainingDataSet.getNumberRows(), trainingDataSet.getNumberCols(), 2, trainingDataSet.getNumberClasses());
            } else if (networkType.equals("mnist_lenet5")) {
                cnn.createLeNet5(ActivationType.LEAKY_RELU5, batchSize, trainingDataSet.getNumberChannels(), trainingDataSet.getNumberRows(), trainingDataSet.getNumberCols(), 2, trainingDataSet.getNumberClasses());
            } else if (networkType.equals("cifar_small_no_pool")) {
                cnn.createSmallNoPool(ActivationType.LEAKY_RELU5, batchSize, trainingDataSet.getNumberChannels(), trainingDataSet.getNumberRows(), trainingDataSet.getNumberCols(), 0, trainingDataSet.getNumberClasses());
            } else if (networkType.equals("cifar_small")) {
                cnn.createSmall(ActivationType.LEAKY_RELU5, batchSize, trainingDataSet.getNumberChannels(), trainingDataSet.getNumberRows(), trainingDataSet.getNumberCols(), 0, trainingDataSet.getNumberClasses());
            } else if (networkType.equals("cifar_lenet5")) {
                cnn.createLeNet5(ActivationType.LEAKY_RELU5, batchSize, trainingDataSet.getNumberChannels(), trainingDataSet.getNumberRows(), trainingDataSet.getNumberCols(), 0, trainingDataSet.getNumberClasses());
            } 
        } catch (NeuralNetworkException e) {
            Log.fatal("ERROR connecting the neural network -- this should not happen!.");
            e.printStackTrace();
            System.exit(1);
        }

        //start the gradient descent
        try {
            Log.info("Starting minibatch gradient descent!");
            Log.info("minibatch (" + batchSize + "), " + dataSetName + ", " + lossFunctionName + ", lr: " + learningRate + ", mu:" + mu);

            cnn.initializeRandomly(initializationType, bias);

            //TODO: For Programming Assignment 2 - Part 3 use this and implement nesterov momentum
            //java will initialize each element in the array to 0
            double[] velocity = new double[cnn.getNumberWeights()];

            double bestError = Double.MAX_VALUE;
            
            Log.info("calculating initial error and accuracy");
            Log.info("bestError error accuracy testingError testingAccuracy");

            double[] accuracyAndError = new double[2];
            cnn.calculateAccuracyAndError(trainingDataSet, batchSize, accuracyAndError);
            double accuracy = accuracyAndError[0];
            double error = accuracyAndError[1];

            cnn.calculateAccuracyAndError(testingDataSet, batchSize, accuracyAndError);
            double testingAccuracy = accuracyAndError[0];
            double testingError = accuracyAndError[1];

            if (error < bestError) bestError = error;
            System.out.println("ITERATION  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make the accuracy a percentage*/ + " " + testingError + String.format("%10.5f", testingAccuracy * 100.0) /* make the test accuracy a percentage */);

            for (int i = 0; i < epochs; i++) {

                //TODO: Programming Assignment 3 - Part 2 you need to implement one epoch (pass through the
                //training data) for minibatch gradient descent with nesterov momemntum
                //For part 3 you will probably want to implement RMSProp or ADAM for the bonus.
                //You can cap the weights at -50 and 50

                trainingDataSet.shuffle();
                cnn.resetRunning();
                for (int j = 0; j < trainingDataSet.getNumberImages(); j += batchSize) {
                    double[] gradient = cnn.getGradient(trainingDataSet, j, batchSize);
                    double[] weights = cnn.getWeights();

                    Vector.copy(velocity, Vector.subtractVector(Vector.multiply(mu, velocity), Vector.multiply(learningRate, gradient)));

                    weights = Vector.addVector(weights, velocity);

                    for (int k = 0; k < weights.length; k++) {
                        if (weights[k] < -50) weights[k] = -50;
                        if (weights[k] > 50) weights[k] = 50;
                    }

                    cnn.setWeights(weights);
                    
                }
                learningRate *= 0.975;
                Log.info("Learning rate: " + learningRate);

                //Log.info("weights: " + Arrays.toString(nn.getWeights()));

                //at the end of each epoch, calculate the error over the entire
                //set of sequences and print it out so we can see if we're decreasing
                //the overall error, also do this for the test data to see how we're
                //doing on unseen data
                cnn.calculateAccuracyAndError(trainingDataSet, batchSize, accuracyAndError);
                accuracy = accuracyAndError[0];
                error = accuracyAndError[1];

                cnn.calculateAccuracyAndError(testingDataSet, batchSize, accuracyAndError);
                testingAccuracy = accuracyAndError[0];
                testingError = accuracyAndError[1];

                if (error < bestError) bestError = error;
                System.out.println("  " + bestError + " " + error + " " + String.format("%10.5f", accuracy * 100.0) /*make the accuracy a percentage*/ + " " + testingError + String.format("%10.5f", testingAccuracy * 100.0) /* make the test accuracy a percentage */);
            }

        } catch (NeuralNetworkException e) {
            Log.fatal("gradient descent failed with exception: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}

