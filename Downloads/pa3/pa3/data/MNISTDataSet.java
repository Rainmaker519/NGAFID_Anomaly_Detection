package data;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import util.Log;

public class MNISTDataSet extends ImageDataSet {

    int bytesToInt(byte[] bytes, int offset) {
        int i = ((bytes[offset]     << 24) & 0xff000000) |
                ((bytes[offset + 1] << 16) & 0x00ff0000) |
                ((bytes[offset + 2] <<  8) & 0x0000ff00) |
                ((bytes[offset + 3] <<  0) & 0x000000ff);

        return i;
    }

    /**
     * Creates a new MNISTDataSet object from an MNIST image file and an MNIST label file 
     *
     * @param imageFilename is the filename of the binary file containing the image data
     * @param labelFilename is the filename of the binary file containing the label data
     * @param expectedImages is the expected number of images in the file (to use as a sanity check)
     *
     */
    public MNISTDataSet(String imageFilename, String labelFilename, int expectedImages) {
        try {
            Log.info("reading image filename '" + imageFilename + "' and label filename: '" + labelFilename);

            Path imagePath = Paths.get(imageFilename);
            byte[] imageBytes = Files.readAllBytes(imagePath);

            Path labelPath = Paths.get(labelFilename);
            byte[] labelBytes = Files.readAllBytes(labelPath);

            //the first four bytes are the "magic number" to make sure you're
            //reading the binary file with the right 'endian'-ness
            //for the image file the magic number is 2051

            int imageMagicNumber = bytesToInt(imageBytes, 0);
            assert imageMagicNumber == 2051 : "Incorrect MNIST image file magic number, should be 2051";

            numberImages = bytesToInt(imageBytes, 4);
            assert numberImages == expectedImages : "Incorrect number of images in MNIST file: " + numberImages + ", should be: " + expectedImages;

            numberRows = bytesToInt(imageBytes, 8);
            assert numberRows == 28 : "Incorrect number of iamges rows in MNIST file: " + numberRows + ", should be: 28";

            numberCols = bytesToInt(imageBytes, 12);
            assert numberCols == 28 : "Incorrect number of iamges cols in MNIST file: " + numberRows + ", should be: 28";

            //the first four bytes are the "magic number" to make sure you're
            //reading the binary file with the right 'endian'-ness
            //for the label file the magic number is 2049
            int labelMagicNumber = bytesToInt(labelBytes, 0);
            assert labelMagicNumber == 2049 : "Incorrect MNIST label file magic number, should be 2049";

            int numberLabels = bytesToInt(imageBytes, 4);
            assert numberLabels == expectedImages : "Incorrect number of labels in MNIST file: " + numberLabels + ", should be: " + expectedImages;

            numberChannels = 1;
            numberClasses = 10;

            //the first 16 bytes (4 ints at 4 bytes per int) were the magic number, number of images, number of rows per image, and number of cols per image. After this each byte is the pixel value for an image (0 - 255)
            int imageOffset = 16;

            //the first 8 bytes (2 ints at 4 bytes per int) were the magic number and number of labels in the label file. After this each byte is the image label.
            int labelOffset = 8;

            for (int i = 0; i < numberImages; i++) {
                Image image = new Image(numberCols, numberRows, imageOffset, imageBytes, labelOffset, labelBytes);
                images.add(image);

                //creating a new image will read one byte per pixel in the image
                imageOffset += (numberRows * numberCols);
                //creating a new Image will read one byte for the label assignment
                labelOffset++;
            }

            Log.info("read " + numberImages + " MNIST images.");
        } catch (IOException e) {
            System.err.println("ERROR reading MNIST files: " + e);
            e.printStackTrace();
        }
    }

    /**
     * Gets the nice human readable name of this DataSet
     * 
     * @return the name of this dataset
     */
    public String getName() {
        return "MNIST";
    }
}
