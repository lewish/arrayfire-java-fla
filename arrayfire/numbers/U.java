package arrayfire.numbers;

public record U(int size) implements IntNumber<U> {
  public U {
    assert size == 1 : "U numbers must always be 1";
  }

  @Override
  public U create(int size) {
    return new U(size);
  }
}
