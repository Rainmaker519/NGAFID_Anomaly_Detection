/**
 * A helpful class of tests to make sure programming assignment 1
 * part 1 is working correctly
 */

import java.util.Arrays;
import java.util.List;

import data.Image;
import data.CIFARDataSet;

import util.Log;
import util.Vector;

public class CIFARTests {

    public static void main(String[] arguments) {
        CIFARDataSet cifarTrain = new CIFARDataSet(new String[]{"./datasets/cifar-10-batches-bin/data_batch_1.bin", "./datasets/cifar-10-batches-bin/data_batch_2.bin", "./datasets/cifar-10-batches-bin/data_batch_3.bin", "./datasets/cifar-10-batches-bin/data_batch_4.bin", "./datasets/cifar-10-batches-bin/data_batch_5.bin"}, 10000);
        for (int i = 0; i < 10; i++) {
            Image image = cifarTrain.images.get(i);
            image.writeJPG("cifar_train_" + i + "_" + image.label + ".jpg");
        }

        CIFARDataSet cifarTest = new CIFARDataSet(new String[]{"./datasets/cifar-10-batches-bin/test_batch.bin"}, 10000);
        for (int i = 0; i < 10; i++) {
            Image image = cifarTest.images.get(i);
            image.writeJPG("cifar_test_" + i + "_" + image.label + ".jpg");
        }
    }
}
