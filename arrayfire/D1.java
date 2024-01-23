package arrayfire;

public record D1(int index) implements Dim {
  public D1 {
    if (index != 1) {
      throw new IllegalArgumentException("index must be 1");
    }
  }

  public D1() {
    this(1);
  }
}