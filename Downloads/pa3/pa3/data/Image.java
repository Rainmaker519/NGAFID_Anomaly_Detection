package data;

import java.io.File;
import java.io.IOException;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

import util.Log;


public class Image {
    //each pixel value is a single byte (between 0 and 255)
    public final int label;
    public final byte[][][] pixels;

    /**
     * This constructs an image from MNIST binary data.
     *
     * @param numberCols is the number of columns in each image (28)
     * @param numberRows is the number of rows in each image (28)
     * @param imageOffset is the index to start reading bytes from imageBytes
     * @param imageBytes are the bytes read from the image binary file
     * @param labelOffset is the index to start reading bytes from labelBytes
     * @param labelBytes are the bytes read from the label binary file
     */
    Image(int numberCols, int numberRows, int imageOffset, byte[] imageBytes, int labelOffset, byte[] labelBytes) {
        pixels = new byte[1][numberCols][numberRows];

        for (int y = 0; y < numberCols; y++) {
            for (int x = 0; x < numberRows; x++) {
                pixels[0][y][x] = imageBytes[imageOffset++];

                //System.out.printf("%4d", Byte.toUnsignedInt(pixels[0][y][x]));
            }
            //System.out.println();
        }

        label = labelBytes[labelOffset];
        //Log.info("label: " + label + "\n");
    }

    /**
     * This constructs an image from CIFAR binary data.
     *
     * @param numberChannels is the number of channels in each image (3)
     * @param numberCols is the number of columns in each image (32)
     * @param numberRows is the number of rows in each image (32)
     * @param offset is the index to start reading bytes
     * @param bytes are the bytes read from the cifar binary file
     */
    Image(int numberChannels, int numberCols, int numberRows, int offset, byte[] bytes) {
        label = bytes[offset];
        offset++;

        pixels = new byte[numberChannels][numberCols][numberRows];

        for (int z =  0; z < numberChannels; z++) {
            for (int y = 0; y < numberCols; y++) {
                for (int x = 0; x < numberRows; x++) {
                    pixels[z][y][x] = bytes[offset++];

                    //System.out.printf("%4d", Byte.toUnsignedInt(pixels[z][y][x]));
                }
                //System.out.println();
            }
        }

        //Log.info("label: " + label + "\n");
    }

    public void writeJPG(String filename) {
        int numberChannels = pixels.length;
        int numberCols = pixels[0].length;
        int numberRows = pixels[0][0].length;

        BufferedImage bufferedImage = new BufferedImage(numberCols, numberRows, BufferedImage.TYPE_INT_RGB);

        if (numberChannels == 3) {
            //this is a CIFAR-10 image, set RGB from each channel

            for (int y = 0; y < numberCols; y++) {
                for (int x = 0; x < numberRows; x++) {
                    //create an int pixel value from the red green and blue channels

                    int a = 255;
                    int r = Byte.toUnsignedInt(pixels[0][y][x]);
                    int g = Byte.toUnsignedInt(pixels[1][y][x]);
                    int b = Byte.toUnsignedInt(pixels[2][y][x]);

                    //set the pixel value
                    int pixel = (a<<24) | (r<<16) | (g<<8) | b;
                    bufferedImage.setRGB(x, y, pixel);
                }
            }
        } else {
            //this is an MNIST image, set r g b from the one channel (so it is B&W)
            for (int y = 0; y < numberCols; y++) {
                for (int x = 0; x < numberRows; x++) {
                    //create an int pixel value from the red green and blue channels

                    int a = 255;
                    int r = Byte.toUnsignedInt(pixels[0][y][x]);
                    int g = r;
                    int b = r;

                    //set the pixel value
                    int pixel = (a<<24) | (r<<16) | (g<<8) | b;
                    bufferedImage.setRGB(x, y, pixel);
                }
            }
        }

        //write image
        try {
            File file = new File(filename);
            ImageIO.write(bufferedImage, "jpg", file);
        } catch(IOException e) {
            System.out.println("Error writing image to file: '" + filename + "'" + e);
            e.printStackTrace();
            System.exit(1);
        }
    }
}
