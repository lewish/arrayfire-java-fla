package arrayfire;

public record D0(int index) implements Dim {
  public D0 {
    if (index != 0) {
      throw new IllegalArgumentException("index must be 0");
    }
  }

  public D0() {
    this(0);
  }
}
