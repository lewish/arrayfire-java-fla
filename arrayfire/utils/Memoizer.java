package arrayfire.utils;

import java.util.function.Supplier;

public class Memoizer {
  public static <T> Supplier<T> memoize(Supplier<T> supplier) {
    return new MemoizedSupplier<>(supplier);
  }

  private static class MemoizedSupplier<T> implements Supplier<T> {
    private final Supplier<T> supplier;
    private T value;

    public MemoizedSupplier(Supplier<T> supplier) {
      this.supplier = supplier;
    }

    @Override
    public T get() {
      if (value == null) {
        value = supplier.get();
      }
      return value;
    }
  }
}