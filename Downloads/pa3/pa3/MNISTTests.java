/**
 * A helpful class of tests to make sure programming assignment 1
 * part 1 is working correctly
 */

import java.util.Arrays;
import java.util.List;

import data.Image;
import data.MNISTDataSet;

import util.Log;
import util.Vector;

public class MNISTTests {

    public static void main(String[] arguments) {
        MNISTDataSet mnistTrain = new MNISTDataSet("./datasets/train-images-idx3-ubyte", "./datasets/train-labels-idx1-ubyte", 60000);
        for (int i = 0; i < 10; i++) {
            Image image = mnistTrain.images.get(i);
            image.writeJPG("mnist_train_" + i + "_" + image.label + ".jpg");
        }


        MNISTDataSet mnistTest = new MNISTDataSet("./datasets/t10k-images-idx3-ubyte", "./datasets/t10k-labels-idx1-ubyte", 10000);
        for (int i = 0; i < 10; i++) {
            Image image = mnistTest.images.get(i);
            image.writeJPG("mnist_test_" + i + "_" + image.label + ".jpg");
        }
    }
}
