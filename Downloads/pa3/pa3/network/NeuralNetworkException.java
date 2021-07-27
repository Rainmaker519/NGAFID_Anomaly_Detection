/**
 * This class is used to throw exceptions related to the
 * NeuralNetwork, Node and Edge classes.
 */
package network;

public class NeuralNetworkException extends Exception {

    /**
     * Creates a new NeuralNetworkException with the specified
     * error message.
     *
     * @param message is the message for this exception.
     */
    public NeuralNetworkException(String message) {
        super(message);
    }

}
