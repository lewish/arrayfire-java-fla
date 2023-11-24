package arrayfire.numbers;

public class IntNumber extends Number {

  private final int value;

  public IntNumber(int value) {
    this.value = value;
  }

  public static U U = new U();

  public static A a(int n) {
    return new A(n);
  }

  public static B b(int n) {
    return new B(n);
  }

  public static N n(int n) {
    return new N(n);
  }

  public int intValue() {
    return value;
  }

  @Override
  public long longValue() {
    return value;
  }

  @Override
  public float floatValue() {
    return (float) value;
  }

  @Override
  public double doubleValue() {
    return value;
  }
}
