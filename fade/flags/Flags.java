package fade.flags;


import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Flags {
  private static boolean parsed = false;
  private static Map<String, String> keyedArgs;


  public static void parse(String... args) {
    parse(false, args);
  }

  public static void parseForTesting(String... args) {
    parse(true, args);
  }

  private static void parse(boolean testing, String... args) {
    if (!testing && parsed) {
      throw new RuntimeException("Already parsed args.");
    }

    var keyedArgs = new HashMap<String, String>();

    String lastKey = null;
    for (int i = 0; i < args.length; i++) {
      var arg = args[i];
      if (arg.startsWith("--")) {
        lastKey = arg.substring(2);
      } else if (lastKey != null) {
        keyedArgs.put(lastKey, arg);
        lastKey = null;
      }
    }

    Flags.keyedArgs = Collections.unmodifiableMap(keyedArgs);
    parsed = true;
  }

  public static Map<String, String> parsedFlags() {
    if (keyedArgs == null) {
      throw new RuntimeException("Flags have not been parsed, call Flags.parse() first.");
    }
    return keyedArgs;
  }

  public static <T> T get(Flag<T> flag) {
    if (keyedArgs == null) {
      if (flag.defaultValue().isPresent()) {
        return flag.defaultValue().get();
      }
      throw new RuntimeException("Flags have not been parsed, call Flags.parse() first.");
    }
    if (keyedArgs.containsKey(flag.name())) {
      return flag.parser().apply(keyedArgs.get(flag.name()));
    }
    return flag.defaultValue().orElseThrow();
  }

  public static Flag<String> stringFlag(String name) {
    return new Flag<String>(name, STRING_PARSER);
  }

  public static Flag<String> stringFlag(String name, String defaultValue) {
    return new Flag<String>(name, STRING_PARSER, defaultValue);
  }

  public static Flag<Long> longFlag(String name) {
    return new Flag<Long>(name, LONG_PARSER);
  }

  public static Flag<Long> longFlag(String name, Long defaultValue) {
    return new Flag<Long>(name, LONG_PARSER, defaultValue);
  }


  public static Flag<Integer> intFlag(String name, Integer defaultValue) {
    return new Flag<>(name, INTEGER_PARSER, defaultValue);
  }

  public static Flag<Double> doubleFlag(String name) {
    return new Flag<>(name, DOUBLE_PARSER);
  }

  public static Flag<Boolean> booleanFlag(String name) {
    return new Flag<>(name, BOOLEAN_PARSER);
  }

  public static Flag<Boolean> booleanFlag(String name, Boolean defaultValue) {
    return new Flag<>(name, BOOLEAN_PARSER, defaultValue);
  }

  public static Flag<Double> doubleFlag(String name, Double defaultValue) {
    return new Flag<>(name, DOUBLE_PARSER, defaultValue);
  }

  public static <E extends Enum<E>> Flag<E> enumFlag(String name, E[] values) {
    return new Flag<E>(name, enumParser(values));
  }

  public static <E extends Enum<E>> Flag<E> enumFlag(String name, E[] values, E defaultValue) {
    return new Flag<E>(name, enumParser(values), defaultValue);
  }

  private static final Function<String, String> STRING_PARSER = Function.identity();
  private static final Function<String, Integer> INTEGER_PARSER = Integer::parseInt;
  private static final Function<String, Long> LONG_PARSER = Long::parseLong;
  private static final Function<String, Double> DOUBLE_PARSER = Double::parseDouble;
  private static final Function<String, Boolean> BOOLEAN_PARSER = Boolean::parseBoolean;

  private static <E extends Enum<E>> Function<String, E> enumParser(E[] values) {
    return Arrays.stream(values).collect(Collectors.toMap(E::name, Function.identity()))::get;
  }
}
