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

public abstract class ImageDataSet {

    public int numberImages;
    public int numberChannels;
    public int numberRows;
    public int numberCols;
    public int numberClasses;

    //all the images in the data set
    public ArrayList<Image> images = new ArrayList<Image>();

    double[] channelAvgs = null;
    double[] channelStdDevs = null;


    /**
     * Specifies what channelAvgs to use for this ImageDataSet, this way
     * we can set them for the test data set from values calculated by
     * the training data set.
     *
     * @param channelAvgs are the channelAvgs calculated by another data set
     */
    public void setChannelAvgs(double[] channelAvgs) {
        this.channelAvgs = channelAvgs;
    }

    /**
     * Specifies what channelStdDevs to use for this ImageDataSet, this way
     * we can set them for the test data set from values calculated by
     * the training data set.
     *
     * @param channelStdDevs are the channelStdDevs calculated by another data set
     */
    public void setChannelStdDevs(double[] channelStdDevs) {
        this.channelStdDevs = channelStdDevs;
    }

    /**
     * Get the average pixel value for each channel
     *
     * @return an array of the average pixel values for each channel
     */
    public double[] getChannelAvgs() {
        //only calculate this once and then re-use it, because we will
        //be getting these on each forward pass

        if (channelAvgs == null) {
        	
        	channelAvgs = new double[numberChannels];
        	
        	for (Image image : images) {
        		for (int channelNum = 0; channelNum < numberChannels; channelNum++) {//we are calculating the average for each channel, this loops through channels
            		for (int x = 0; x < numberCols; x++) {
            			for (int y = 0; y < numberRows; y++) {
            				int toAdd = Byte.toUnsignedInt(image.pixels[channelNum][x][y]);
            				channelAvgs[channelNum] += toAdd;
            			}
            		}
            	}
        	}
        	
        	for (int i = 0; i < channelAvgs.length; i++) {
        		channelAvgs[i] /= images.size() * (numberRows * numberCols);
        	}
        	
        }
		
        return channelAvgs;
    }

    /**
     * Returns the standard devation for each pixel in every image for each column.
     *
     * @return an array of the minimum values for each time series column across all time series
     */
    public double[] getChannelStdDevs(double[] avgs) {
        //only calculate this once and then re-use it, because we will
        //be getting these on each forward pass

        if (channelStdDevs == null) {
            //TODO: You need to implement this for Programming Assignment 3 - Part 1
            //You need to calculate the standard deviation of each channel (there is only one 
            //channel for CIFAR but 3 for CIFAR-10), and this will require
            //the previously computed averages. set the classes's
            //channelStdDevs here so it isn't recalculated
            //Make sure you use Byte.toUnsignedInt to convert the bytes
        	double[] channelAvgs = getChannelAvgs();
        	double[] channelStd = new double[numberChannels];
        	
        	for (Image image : images) {
        		for (int channelNum = 0; channelNum < numberChannels; channelNum++) {//we are calculating the average for each channel, this loops through channels
            		for (int x = 0; x < numberCols; x++) {
            			for (int y = 0; y < numberRows; y++) {
            				int toAdd = Byte.toUnsignedInt(image.pixels[channelNum][x][y]);
            				channelStd[channelNum] += (toAdd - channelAvgs[channelNum]) * (toAdd - channelAvgs[channelNum]);
            			}
            		}
        		}
        	}
        	for (int i = 0; i < channelStd.length; i++) {
        		channelStd[i] /= images.size() * (numberRows * numberCols);
        		channelStd[i] = Math.sqrt(channelStd[i]);
        	}
        	
        	channelStdDevs = channelStd;
        }

        return channelStdDevs;
    }

    /**
     * Gets the nice human readable name of this DataSet
     * 
     * @return the name of this dataset
     */
    public abstract String getName();

    /**
     * Gets the number of images in this DataSet
     *
     * @return the number of images in this DataSet
     */
    public int getNumberImages() {
        return numberImages;
    }

    /**
     * Gets the number of channels in an image in this dataset
     *
     * @return the number of channels in each image
     */
    public int getNumberChannels() {
        return numberChannels;
    }


    /**
     * Gets the number of rows in an image in this dataset
     *
     * @return the number of rows in each image
     */
    public int getNumberRows() {
        return numberRows;
    }

    /**
     * Gets the number of cols in an image in this dataset
     *
     * @return the number of cols in each image
     */
    public int getNumberCols() {
        return numberCols;
    }

    /**
     * Gets the number of classes in an image in this dataset
     *
     * @return the number of classes in each image
     */
    public int getNumberClasses() {
        return numberClasses;
    }



    /**
     * This randomly shuffles the orders of the images in the 
     * images ArrayList. This will be useful when we are implementing 
     * different versions of stochastic backpropagation.
     */
    public void shuffle() {
        Log.trace("Shuffling '" + getName() + "'");
        Collections.shuffle(images);
    }

    /**
     * This reduces the number of images in the dataset so the tests
     * don't take as long.
     */
    public void resize(int newSize) {
        //shuffle();
        images.subList(newSize, images.size()).clear();
        Log.info("Images size now: " + images.size());
        numberImages = images.size();
    }


    /**
     * This gets a consecutive set of images from the images
     * ArrayList. position should be >= 0 and numberOfImages should
     * be >= 1.
     *
     * @param position the position of the first images to return
     * @param numberOfImages is how many images to return. If 
     * position + numberOfImages is > than images.size() it will
     * return the remaining images in the images ArrayList.
     *
     * @return An ArrayList of the images specified by position and
     * numberOfImages. Its size will be <= numberOfImages.
     */
    public List<Image> getImages(int position, int numberOfImages) {
        int endIndex = position + numberOfImages;
        if (endIndex > images.size()) endIndex = images.size();

        Log.trace("Getting images[" + position + " to " + endIndex + "] from 'CIFAR'");
            
        List<Image> subList = images.subList(position, endIndex);
        return subList;
    }

}
