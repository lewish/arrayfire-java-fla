package arrayfire.utils;

import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

public final class NamedContextual<T> implements Contextual<T> {
  private final String name;
  private final Supplier<Optional<T>> defaultValue;

  public NamedContextual(String name, Supplier<Optional<T>> defaultValue) {
    this.name = name;
    this.defaultValue = defaultValue;
  }

  @Override
  public T get() {
    return Context.get(this);
  }

  @Override
  public Context.Entry<T> create(T value) {
    return new Context.Entry<>(this, value);
  }

  public String name() {
    return name;
  }

  @Override
  public Optional<T> defaultValue() {
    return defaultValue.get();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this)
      return true;
    if (obj == null || obj.getClass() != this.getClass())
      return false;
    var that = (NamedContextual) obj;
    return Objects.equals(this.name, that.name) && Objects.equals(this.defaultValue, that.defaultValue);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, defaultValue);
  }

  @Override
  public String toString() {
    return "NamedContextual[" + "name=" + name + ", " + "defaultValue=" + defaultValue + ']';
  }

}
