package arrayfire;

public record Version(int major, int minor, int patch) {
  @Override
  public String toString() {
    return major + "." + minor + "." + patch;
  }
}
