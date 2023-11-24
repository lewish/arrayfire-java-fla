package fade.context;

import java.util.Optional;
import java.util.function.Supplier;

public interface Contextual<T> {
  Optional<T> defaultValue();

  T get();

  Context.Entry<T> create(T value);


  static <T> Contextual<T> named(String name, Supplier<T> defaultValue) {
    return new NamedContextual<>(name, () -> Optional.of(defaultValue.get()));
  }

  static <T> Contextual<T> named(String name, T defaultValue) {
    return new NamedContextual<>(name, () -> Optional.of(defaultValue));
  }

  static <T> Contextual<T> named(String name) {
    return new NamedContextual<>(name, Optional::empty);
  }
}
