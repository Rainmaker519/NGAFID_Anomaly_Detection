
import java.util.Arrays;
import java.util.Random;


public class NormTest {
    
    public static void main(String[] arguments) {
        Random generator = new Random();

        double[] v = new double[20];

        double originalNorm = 0.0;
        for (int i = 0; i < v.length; i++) {
            v[i] = generator.nextDouble() * 20 - 10;

            originalNorm += v[i] * v[i];
        }

        originalNorm = Math.sqrt(originalNorm);

        System.out.println(Arrays.toString(v));
        System.out.println("norm: " + originalNorm);

        double cap = 3;
        System.out.println("capping at: " + cap);

        double[] desired = new double[v.length];
        double desiredNorm = 0.0;
        for (int i = 0; i < v.length; i++) {
            if (v[i] > cap) desired[i] = cap;
            else if (v[i] < -cap) desired[i] = -cap;
            else desired[i] = v[i];


            desiredNorm += desired[i] * desired[i];
        }
        desiredNorm = Math.sqrt(desiredNorm);

        System.out.println(Arrays.toString(desired));
        System.out.println("desiredNorm: " + desiredNorm);

        double[] output = new double[v.length];
        double newNorm = 0.0;
        for (int i = 0; i < v.length; i++) {
            output[i] = v[i] * (desired[i] / (0.00001 + originalNorm));
            newNorm += output[i] * output[i];
        }
        newNorm = Math.sqrt(newNorm);

        System.out.println(Arrays.toString(output));
        System.out.println("newNorm: " + newNorm);
    }
}
