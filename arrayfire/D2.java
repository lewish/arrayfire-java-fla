package arrayfire;

public record D2(int index) implements Dim {
  public D2 {
    if (index != 2) {
      throw new IllegalArgumentException("index must be 2");
    }
  }

  public D2() {
    this(2);
  }
}