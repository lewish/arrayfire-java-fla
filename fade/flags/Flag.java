package fade.flags;

import java.util.Optional;
import java.util.function.Function;

public record Flag<T>(String name, Function<String, T> parser, Optional<T> defaultValue) {

  public Flag(String name, Function<String, T> parser, T value) {
    this(name, parser, Optional.of(value));
  }

  public Flag(String name, Function<String, T> parser) {
    this(name, parser, Optional.empty());
  }

  public T get() {
    return Flags.get(this);
  }
}
