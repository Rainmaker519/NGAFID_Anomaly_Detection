/**
 * This class represents an ConvolutionalEdge in a neural network. It will contain
 * the ConvolutionalEdge's weight, and also have references to input node and output
 * nodes of this edge.
 */
package network;

import java.util.Random;

import util.Log;

public abstract class Edge {
    //the z dimension size for this filter
    int sizeZ;

    //the y dimension size for this filter
    int sizeY;

    //the x dimension size for this filter
    int sizeX;

    //the input node of this edge
    public ConvolutionalNode inputNode;

    //the output node of this edge
    public ConvolutionalNode outputNode;

    /**
     * This constructs a new edge in the neural network between the passed
     * parameters. It will register itself at the input and output nodes
     * through the Node.addOutgoingConvolutionalEdge(ConvolutionalEdge) and Node.addIncomingConvolutionalEdge(ConvolutionalEdge)
     * methods.
     *
     * @param inputNode is the input for this edge
     * @param outputNode is the output for this edge
     */
    public Edge(ConvolutionalNode inputNode, ConvolutionalNode outputNode, int sizeZ, int sizeY, int sizeX) throws NeuralNetworkException {
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        this.sizeZ = sizeZ;
        this.sizeY = sizeY;
        this.sizeX = sizeX;

        Log.trace("Created a new " + this.getClass().toString() + " with input " + inputNode.toString() + " and output " + outputNode.toString());

        inputNode.addOutgoingEdge(this);
        outputNode.addIncomingEdge(this);
    }

    /**
     * Resets all the deltas for this edge
     */
    public abstract void reset();

    /**
     * Used to get the weights of this Edge along with the weights.
     * It will set the weights in the weights
     * parameter passed in starting at position, and return the number of
     * weights it set.
     *
     * @param position is the index to start setting weights in the weights parameter
     * @param weights is the array of weights we're setting.
     *
     * @return the number of weights set in the weights parameter
     */
    public abstract int getWeights(int position, double[] weights);

    /**
     * Used to get the deltas of this Edge along with the deltas.
     * It will set the deltas in the deltas
     * parameter passed in starting at position, and return the number of
     * deltas it set.
     *
     * @param position is the index to start setting deltas in the deltas parameter
     * @param deltas is the array of deltas we're setting.
     *
     * @return the number of deltas set in the deltas parameter
     */
    public abstract int getDeltas(int position, double[] deltas);

    /**
     * Used to set the weights of this edge .
     * It uses the same technique as Node.getWeights
     * where the starting position of weights to set is passed, and it returns
     * how many weights were set.
     * 
     * @param position is the starting position in the weights parameter to start
     * setting weights from.
     * @param weights is the array of weights we are setting from
     *
     * @return the number of weights gotten from the weights parameter
     */
    public abstract int setWeights(int position, double[] weights);

    /**
     * Used to print gradients related to this edge, along with informationa
     * about this edge.
     * It start printing the gradients passed in starting at position, and 
     * return the number of gradients it printed.
     *
     * @param position is the index to start printing different gradients
     * @param numericGradient is the array of the numeric gradient we're printing
     * @param backpropGradient is the array of the backprop gradient we're printing
     *
     * @return the number of gradients printed by this edge
     */
    public abstract int printGradients(int position, double[] numericGradient, double[] backpropGradient);


    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public abstract void propagateForward(double[][][][] inputValues);

    /**
     * This takes an incoming delta from the output node
     * and propagates it backwards to the input node.
     *
     * @param delta is the delta/error from the output node.
     */
    public abstract void propagateBackward(double[][][][] delta);

    /**
     * Compares if two ConvolutionalEdge objects are equal. Two recurrent edges are the same
     * if they are from the same input node to the same output node and their time skip is
     * the same.
     *
     * @return true if the two recurrent edge objects have the same input and output nodes and the same time skip
     */
    public boolean equals(ConvolutionalEdge other) {
        if (other.inputNode == inputNode && other.outputNode == outputNode) return true;
        return false;
    }
}
