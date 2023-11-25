package arrayfire.utils;

public class Functions {

  public static <A, R> Function<A, R> function(Function<A, R> func) {
    return func;
  }

  public static <A, B, R> Function2<A, B, R> function2(Function2<A, B, R> func) {
    return func;
  }

  public static <A, B, C, R> Function3<A, B, C, R> function3(Function3<A, B, C, R> func) {
    return func;
  }

  @FunctionalInterface
  public interface Function<A, R> {
    R apply(A a);
  }

  @FunctionalInterface
  public interface Function2<A, B, R> {
    R apply(A a, B b);
  }

  @FunctionalInterface
  public interface Function3<A, B, C, R> {
    R apply(A a, B b, C c);
  }

  @FunctionalInterface
  public interface Consumer<A> {
    void consume(A a);
  }

  @FunctionalInterface
  public interface Consumer2<A, B> {
    void consume(A a, B b);
  }
}
