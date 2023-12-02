package arrayfire;

public class ArrayFireException extends RuntimeException {

  private final Status status;

  public ArrayFireException(Status status) {
    super(String.format("ArrayFireException: %S", status.name()));
    this.status = status;
  }

  public Status status() {
    return status;
  }
}
