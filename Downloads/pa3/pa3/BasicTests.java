/**
 * A helpful class of tests to make sure programming assignment 1
 * part 1 is working correctly
 */

import java.util.Arrays;
import java.util.List;

import network.NeuralNetworkException;

import util.Log;
import util.Vector;

public class BasicTests {

    /**
     * When comparing the weights and outputs of a neural network they might be
     * slightly different when calculated by the numeric gradient and backprop. Also
     * different operating systems and java versions may have exp, log and pow functions
     * implemnented slightly different. Even further, the order of operations with double
     * precision values can slightly change outputs, so when checking equality to the tests
     * we just want to make sure it's close enough.
     */
    public static boolean closeEnough(double n1, double n2) {
        return Math.abs(n1 - n2) < 2e-6;
    }

    /**
     * Use the BasicTestscloseEnough(double, double) method over two
     * arrays to determine if all array values are close enough.
     */
    public static boolean closeEnough(double[] v1, double[] v2) {
        for (int i = 0; i < v1.length; i++) {
            //if any pair of elements aren't close enough the arrays
            //aren't close enough
            if (!closeEnough(v1[i], v2[i])) return false;
        }

        //if we reached here both were close enough
        return true;
    }

    /**
     * When comparing gradients we want to use a relative strategy, as for example if the gradients are close to 10
     * and they are off by 1e-5 it is not very bad, but if the gradients are close to 1e-4
     * and off by 1e-5 then the difference is much more significant.
     *
     * In general a relativeError > 1e-2 is a problem.
     * 1e-2 >= relativeError >= 1e-4 is not very good and indicative of a problem.
     * 1e-4 >= relativeError is good if your objective has a kink in it, not so good otherwise
     * (for now we're using tanh and sigmoid so this is not so good)
     * 1e-7 >= relative error is good
     */
    public static boolean gradientsCloseEnough(double[] g1, double[] g2) {
        if (Vector.norm(g1) == 0.0 && Vector.norm(g2) == 0.0) return true;

        double relativeError = Vector.norm(Vector.subtractVector(g1, g2)) / Math.max(Vector.norm(g1), Vector.norm(g2));
        Log.info("norm g1: " + Vector.norm(g1) + ", norm g2: " + Vector.norm(g2) + ", normDifference: " + Vector.norm(Vector.subtractVector(g1, g2)));

        /*
           double numerator = 0.0;
           double denominator = 0.0;
           double relativeError = 0.0;
           int zeroCount = 0;
           for (int i = 0; i < g1.length; i++) {
           numerator = Math.abs(g1[i] - g2[i]);
           denominator = Math.max(Math.abs(g1[i]), Math.abs(g2[i]));

           if (denominator == 0) {
           zeroCount++;
           } else {
           relativeError += numerator / denominator;
           }
           }

           Log.info("total relative error: " + relativeError + ", g1.length: " + g1.length + ", zeroCount: " + zeroCount);

           relativeError /= (g1.length - zeroCount);
       */

        if (relativeError >= 1e-4) {
            Log.error("relativeError bad: " + relativeError);
            for (int i = 0; i < g1.length; i++) {
                Log.error("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]));
            }

        } else if (relativeError >= 1e-5) {
            Log.warning("relativeError probably bad: " + relativeError);
            for (int i = 0; i < g1.length; i++) {
                Log.trace("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]));
            }

        } else if (relativeError >= 1e-7) {
            Log.debug("relativeError might be bad: " + relativeError);
            for (int i = 0; i < g1.length; i++) {
                Log.trace("\tg1[" + i + "]: " + g1[i] + ", g2[" + i + "]: " + g2[i] + ", difference: " + Math.abs(g1[i] - g2[i]));
            }
        }

        Log.info("relativeError: " + relativeError);
        return relativeError <= 1e-4;
    }

}
