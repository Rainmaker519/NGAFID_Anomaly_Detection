package util;


public class Vector {
    /**
     * Subtracts the second vector from the first. Both vectors should have the
     * same number of elements.
     *
     * @param v1 is the first vector
     * @param v2 is the second vector
     *
     * @return v1 - v2
     */
    public static double[] subtractVector(double[] v1, double[] v2) {
        double[] result = new double[v1.length];

        for (int i = 0; i < v1.length; i++) {
            result[i] = v1[i] - v2[i];
        }

        return result;
    }

    /**
     * Adds the second vector from the first. Both vectors should have the
     * same number of elements.
     *
     * @param v1 is the first vector
     * @param v2 is the second vector
     *
     * @return v1 - v2
     */
    public static double[] addVector(double[] v1, double[] v2) {
        double[] result = new double[v1.length];

        for (int i = 0; i < v1.length; i++) {
            result[i] = v1[i] + v2[i];
        }

        return result;
    }


    /**
     * Calculates the L1 norm of the vector (i.e., the sum of all the absolute values).
     *
     * @param v is the vector
     *
     * @return the L1 norm of the vector
     */
    public static double norm(double[] v) {
        double l1 = 0.0;
        for (int i = 0; i < v.length; i++) {
            l1 += Math.abs(v[i]);
        }
        return l1;
    }

    /**
     * Multiplies the vector by a particular value
     *
     * @param scale is the value the vector is being multiplied
     * @param v is the vector
     *
     * @return a new vector equal to scale * v
     */
    public static double[] multiply(double scale, double[] v) {
        double[] result = new double[v.length];

        for (int i = 0; i < v.length; i++) {
            result[i] = v[i] * scale;
        }

        return result;
    }

    /**
     * Copies the source vector into the target vector
     *
     * @param source is the vector to be copied from
     * @param target is the vector to be copied into
     */
    public static void copy(double[] target, double[] source) {
        for (int i = 0; i < source.length; i++) {
            target[i] = source[i];
        }
    }


    public static void print(double[] v) {
        for (int i = 0; i < v.length; i++) {
            if (i > 0) System.out.print(", ");
            System.out.print(v[i]);
        }
        System.out.println();
    }
}
