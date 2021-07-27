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

public class CIFARDataSet extends ImageDataSet {

    /**
     * Creates a new CIFARDataSet object from an array of cifar-10 binary filenames
     *
     * @param imageFilenames is an array of binary cifar-10 filenames
     * @param expectedImages is the expected number of images in each file (as a sanity check)
     *
     */
    public CIFARDataSet(String[] imageFilenames, int expectedImages) {
        numberChannels = 3;
        numberCols = 32;
        numberRows = 32;
        numberClasses = 10;

        try {
            for (int i = 0; i < imageFilenames.length; i++) {

                Log.info("reading imageFilenames[" + i + "]: '" + imageFilenames[i]);

                Path imagePath = Paths.get(imageFilenames[i]);
                byte[] imageBytes = Files.readAllBytes(imagePath);

                //the cifar image files first contain a 4 byte label, and then the pixel
                //values by channel, column and row
                int offset = 0;

                for (int j = 0; j < expectedImages; j++) {
                    Image image = new Image(numberChannels, numberCols, numberRows, offset, imageBytes);
                    images.add(image);

                    offset += 1 + (numberChannels * numberCols * numberRows);
                }
                Log.info("read " + expectedImages + " CIFAR images from imageFilenames[" + i + "]: '" + imageFilenames[i] + "'.");
            }

            Log.info("read " + images.size() + " CIFAR images.");
            numberImages = images.size();

        } catch (IOException e) {
            System.err.println("ERROR reading CIFAR files: " + e);
            e.printStackTrace();
        }
    }

    /**
     * Gets the nice human readable name of this DataSet
     * 
     * @return the name of this dataset
     */
    public String getName() {
        return "CIFAR";
    }
}
