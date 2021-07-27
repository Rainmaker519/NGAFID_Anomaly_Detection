/**
 * This is a helpful class to make printing out messages easier. You can specify
 * at the command line with the -DLOG_LEVEL=<LEVEL> system property what levels of
 * messages to print out. For example, if you do -DLOG_LEVEL=WARNING, it will print out
 * messages that are at the FATAL, ERROR and WARNING messages but nothing higher. By
 * default the level is INFO, so INFO, WARNING, ERROR and FATAL messages are printed out.
 */
package util;

public class Log {
    private static String[] levels = {"NONE", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE", "ALL"};
    private static final int NONE = 0;
    private static final int FATAL = 1;
    private static final int ERROR = 2;
    private static final int WARNING = 3;
    private static final int INFO = 4;
    private static final int DEBUG = 5;
    private static final int TRACE = 6;
    private static final int ALL = 7;

    private static int level = Log.INFO;

    /**
     * Initialize the Log on startup, logging level is set to INFO by default, otherwise
     * it is read from the "LOG_LEVEL" system property 
     */
    static {
        String logLevelStr = System.getProperty("LOG_LEVEL");
        if (logLevelStr != null) {
            if (logLevelStr.equals("0") || logLevelStr.toLowerCase().equals("none")) level = Log.NONE;
            if (logLevelStr.equals("1") || logLevelStr.toLowerCase().equals("fatal")) level = Log.FATAL;
            if (logLevelStr.equals("2") || logLevelStr.toLowerCase().equals("error")) level = Log.ERROR;
            if (logLevelStr.equals("3") || logLevelStr.toLowerCase().equals("warning")) level = Log.WARNING;
            if (logLevelStr.equals("4") || logLevelStr.toLowerCase().equals("info")) level = Log.INFO;
            if (logLevelStr.equals("5") || logLevelStr.toLowerCase().equals("debug")) level = Log.DEBUG;
            if (logLevelStr.equals("6") || logLevelStr.toLowerCase().equals("trace")) level = Log.TRACE;
            if (logLevelStr.equals("7") || logLevelStr.toLowerCase().equals("all")) level = Log.ALL;
        }

        System.out.println("Log level set to " + levels[level]);
    }

    /**
     * Used when comparing numeric to backprop gradients to help print
     * out a nice error message.
     *
     * @param numeric is the numeric gradient
     * @param backprop is the gradient calculated by backpropagation
     */
    public static String twoGradients(double numeric, double backprop) {
        String output = "numeric: " + numeric + ", backprop: " + backprop;

        double diff = numeric - backprop;
        double max = Math.max(numeric, backprop);

        if (Math.abs(diff/max) > 1e-6) {
            output += " -- DO NOT MATCH, diff: " + diff;
        }

        return output;
    }

    /**
     * Writes a message to the log if Log.level is greater than or equal to Log.FATAL
     * @param message is the message to write to the log
     */
    public static void fatal(String message) {
        if (Log.FATAL <= level) System.out.println("[FATAL  ] " + message);
    }

    /**
     * Writes a message to the log if Log.level is greater than or equal to Log.ERROR
     * @param message is the message to write to the log
     */
    public static void error(String message) {
        if (Log.ERROR <= level) System.out.println("[ERROR  ] " + message);
    }

    /**
     * Writes a message to the log if Log.level is greater than or equal to Log.WARNING
     * @param message is the message to write to the log
     */
    public static void warning(String message) {
        if (Log.WARNING <= level) System.out.println("[WARNING] " + message);
    }

    /**
     * Writes a message to the log if Log.level is greater than or equal to Log.INFO
     * @param message is the message to write to the log
     */
    public static void info(String message) {
        if (Log.INFO <= level) System.out.println("[INFO   ] " + message);
    }

    /**
     * Writes a message to the log if Log.level is greater than or equal to Log.DEBUG
     * @param message is the message to write to the log
     */
    public static void debug(String message) {
        if (Log.DEBUG <= level) System.out.println("[DEBUG  ] " + message);
    }

    /**
     * Writes a message to the log if Log.level is greater than or equal to Log.TRACE
     * @param message is the message to write to the log
     */
    public static void trace(String message) {
        if (Log.TRACE <= level) System.out.println("[TRACE  ] " + message);
    }

}
