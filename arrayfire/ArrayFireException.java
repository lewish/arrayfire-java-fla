package arrayfire;

public class ArrayFireException extends RuntimeException {

  private final AfStatus status;

  public ArrayFireException(AfStatus status) {
    super(String.format("ArrayFireException: %S", status.name()));
    this.status = status;
  }

  public AfStatus status() {
    return status;
  }
}
