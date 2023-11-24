package arrayfire.dims;

public record D3(int index) implements Dim {
  public D3 {
    if (index != 3) {
      throw new IllegalArgumentException("index must be 3");
    }
  }

  public D3() {
    this(3);
  }
}