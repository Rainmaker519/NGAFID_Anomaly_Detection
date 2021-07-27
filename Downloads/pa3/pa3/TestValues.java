/**
 * Use this to store the large arrays and loss function values to compare
 * correctness of forward passes
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import java.util.Arrays;
import java.util.List;

import network.NeuralNetworkException;

import util.Log;


public class TestValues {
    public static final double difference_limit = 1e-6;



    public static double readValue(String name, int seed) {
        String valueFilename = "./test_values/value_" + name + "_" + seed;
        valueFilename = valueFilename.replace(' ', '_');

        Log.debug("Reading from value test file: '" + valueFilename + "'" );

        double value = -1;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(valueFilename)));

            String line = br.readLine();

            value = Double.parseDouble(line);
        } catch (IOException e) {
            Log.fatal("Error reading from value test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from value test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return value;
    }

    public static void writeValue(double value, String name, int seed) {
        String valueFilename = "./test_values/value_" + name + "_" + seed;
        valueFilename = valueFilename.replace(' ', '_');

        Log.debug("Writing to value test file: '" + valueFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(valueFilename)));

            br.write(String.valueOf(value));

        } catch (IOException e) {
            Log.fatal("Error reading from value test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from value test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
    }

    /**
     * Used to check if two values are close enough.
     */
    public static boolean testValue(double n1, double n2, String name, int seed) throws NeuralNetworkException {
        if (Math.abs(n1 - n2) > difference_limit) {
            throw new NeuralNetworkException(name + " for seed " + seed + " calculated value: " + n1 + " was not close enough to precomputed value: " + n2);
        } else {
            return true;
        }
    }

    public static double[] readArray1d(String arrayName, int seed) {
        String arrayFilename = "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\test_values\\" + arrayName + "_" + seed;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Reading from " + arrayName + " test file: '" + arrayFilename + "'" );

        double[] array = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            String line = br.readLine();
            int d1 = Integer.parseInt(line);
            //read each value of the array from its own line in order
            array = new double[d1];
            for (int i = 0; i < d1; i++) {
                line = br.readLine();
                array[i] = Double.parseDouble(line);
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return array;
    }


    public static void writeArray1d(double[] array, String arrayName, int seed) {
        String arrayFilename = "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\test_values\\" + arrayName + "_" + seed;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Writing to " + arrayName + " test file: '" + arrayFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            br.write(String.valueOf(array.length));
            br.write("\n");

            //write each value of the array on its own line in order
            for (int i = 0; i < array.length; i++) {
                br.write(String.valueOf(array[i]));
                br.write("\n");
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
     }

    /**
     * Compare if two arrays are close enough to be the same
     */
    public static boolean testArray1d(double[] m1, double[] m2, String arrayName, int seed) throws NeuralNetworkException {
        if (m1.length != m2.length) {
            throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed was " + m2.length + " and calculated was : " + m1.length);
        }

        for (int i = 0; i < m1.length; i++) {
            if (Math.abs(m1[i] - m2[i]) > difference_limit) {
                throw new NeuralNetworkException(arrayName + " for seed " + seed + " outputs were not the same at index[" + i + "] calculated: " + m1[i] + " vs. precomputed: " + m2[i]);
            }
        }

        return true;
    }


    public static double[][] readArray2d(String arrayName, int seed) {
        String arrayFilename = "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\test_values\\" + arrayName + "_" + seed;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Reading from " + arrayName + " test file: '" + arrayFilename + "'" );

        double[][] array = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            String line = br.readLine();
            int d1 = Integer.parseInt(line);

            //the first line is the second dimension of the output values array
            line = br.readLine();
            int d2 = Integer.parseInt(line);

            //read each value of the array from its own line in order
            array = new double[d1][d2];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    line = br.readLine();
                    array[i][j] = Double.parseDouble(line);
                }
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return array;
    }


    public static void writeArray2d(double[][] array, String arrayName, int seed) {
        String arrayFilename = "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\test_values\\" + arrayName + "_" + seed;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Writing to " + arrayName + " test file: '" + arrayFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            br.write(String.valueOf(array.length));
            br.write("\n");

            //the first line is the second dimension of the output values array
            br.write(String.valueOf(array[0].length));
            br.write("\n");

            //write each value of the array on its own line in order
            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    br.write(String.valueOf(array[i][j]));
                    br.write("\n");
                }
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
     }

    /**
     * Compare if two arrays are close enough to be the same
     */
    public static boolean testArray2d(double[][] m1, double[][] m2, String arrayName, int seed) throws NeuralNetworkException {
        if (m1.length != m2.length) {
            throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed was " + m2.length + " and calculated was : " + m1.length);
        }

        for (int i = 0; i < m1.length; i++) {
            if (m1[i].length != m2[i].length) {
                throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed[" + i + "] was " + m2[i].length + " and calculated[" + i + "] was : " + m1[i].length);
            } 

            for (int j = 0; j < m1[i].length; j++) {
                if (Math.abs(m1[i][j] - m2[i][j]) > difference_limit) {
                    throw new NeuralNetworkException(arrayName + " for seed " + seed + " outputs were not the same at index[" + i + "][" + j + "] calculated: " + m1[i][j] + " vs. precomputed: " + m2[i][j]);
                }
            }
        }

        return true;
    }

    public static double[][][][] readArray4d(String arrayName, int seed) {
        String arrayFilename = "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\test_values\\" + arrayName + "_" + seed;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Reading from " + arrayName + " test file: '" + arrayFilename + "'" );

        double[][][][] array = null;

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            String line = br.readLine();
            int d1 = Integer.parseInt(line);

            //the second line is the second dimension of the output values array
            line = br.readLine();
            int d2 = Integer.parseInt(line);

            //the third line is the third dimension of the output values array
            line = br.readLine();
            int d3 = Integer.parseInt(line);

            //the fourth line is the fourth dimension of the output values array
            line = br.readLine();
            int d4 = Integer.parseInt(line);

            //read each value of the array from its own line in order
            array = new double[d1][d2][d3][d4];
            for (int i = 0; i < d1; i++) {
                for (int j = 0; j < d2; j++) {
                    for (int k = 0; k < d3; k++) {
                        for (int l = 0; l < d4; l++) {
                            line = br.readLine();
                            array[i][j][k][l] = Double.parseDouble(line);
                        }
                    }
                }
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }

        return array;
    }


    public static void writeArray4d(double[][][][] array, String arrayName, int seed) {
        String arrayFilename = "C:\\Users\\Charlie\\Downloads\\pa3\\pa3\\test_values\\" + arrayName + "_" + seed;
        arrayFilename = arrayFilename.replace(' ', '_');

        Log.debug("Writing to " + arrayName + " test file: '" + arrayFilename + "'" );

        BufferedWriter br = null;
        try {
            br = new BufferedWriter(new FileWriter(new File(arrayFilename)));

            //the first line is the first dimension of the output values array
            br.write(String.valueOf(array.length));
            br.write("\n");

            //the second line is the second dimension of the output values array
            br.write(String.valueOf(array[0].length));
            br.write("\n");

            //the third line is the third dimension of the output values array
            br.write(String.valueOf(array[0][0].length));
            br.write("\n");

            //the fourth line is the fourth dimension of the output values array
            br.write(String.valueOf(array[0][0][0].length));
            br.write("\n");

            //write each value of the array on its own line in order
            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    for (int k = 0; k < array[0][0].length; k++) {
                        for (int l = 0; l < array[0][0][0].length; l++) {
                            br.write(String.valueOf(array[i][j][k][l]));
                            br.write("\n");
                        }
                    }
                }
            }

        } catch (IOException e) {
            Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                Log.fatal("Error reading from " + arrayName + " test file, this should not happen!");
                e.printStackTrace();
                System.exit(1);
            }
        }
     }

    /**
     * Compare if two arrays are close enough to be the same
     */
    public static boolean testArray4d(double[][][][] m1, double[][][][] m2, String arrayName, int seed) throws NeuralNetworkException {
        if (m1.length != m2.length) {
            throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed was " + m2.length + " and calculated was : " + m1.length);
        }

        for (int i = 0; i < m1.length; i++) {
            if (m1[i].length != m2[i].length) {
                throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed[" + i + "] was " + m2[i].length + " and calculated[" + i + "] was : " + m1[i].length);
            } 

            for (int j = 0; j < m1[i].length; j++) {
                if (m1[i][j].length != m2[i][j].length) {
                    throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed[" + i + "][" + j + "] was " + m2[i][j].length + " and calculated[" + i + "][" + j + "] was : " + m1[i][j].length);
                } 

                for (int k = 0; k < m1[i][j].length; k++) {
                    if (m1[i][j][k].length != m2[i][j][k].length) {
                        throw new NeuralNetworkException(arrayName + " for seed " + seed + " dimensions were not okay. Pre-computed[" + i + "][" + j + "][" + k + "] was " + m2[i][j][k].length + " and calculated[" + i + "][" + j + "][" + k + "] was : " + m1[i][j][k].length);
                    } 

                    for (int l = 0; l < m1[i][j][k].length; l++) {

                        if (Math.abs(m1[i][j][k][l] - m2[i][j][k][l]) > difference_limit) {
                            throw new NeuralNetworkException(arrayName + " for seed " + seed + " outputs were not the same at index[" + i + "][" + j + "][" + k + "][" + l +"] calculated: " + m1[i][j][k][l] + " vs. precomputed: " + m2[i][j][k][l]);
                        }
                    }
                }
            }
        }

        return true;
    }

    public static void printArray4d(String name, double[][][][] m) {
        System.out.println(name + ":");

        for (int i = 0; i < m.length; i++) {
            System.out.println("\tbatch: " + i);

            for (int z = 0; z < m[i].length; z++) {
                System.out.println("\t\tchannel: " + z);

                for (int y = 0; y < m[i][z].length; y++) {
                    System.out.print("\t\t\t");
                    for (int x = 0; x < m[i][z][y].length; x++) {
                        System.out.printf("%15.10f", m[i][z][y][x]);
                    }
                    System.out.println();
                }
            }
        }
    }

}
