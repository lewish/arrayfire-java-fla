package arrayfire.dims;

public record D3(int index) implements Dim {
  public static final int INDEX = 3;

  public D3 {
    if (index != INDEX) {
      throw new IllegalArgumentException("index must be " + INDEX);
    }
  }

  public D3() {
    this(INDEX);
  }
}