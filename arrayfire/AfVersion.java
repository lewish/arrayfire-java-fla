package arrayfire;

public record AfVersion(int major, int minor, int patch) {
  @Override
  public String toString() {
    return major + "." + minor + "." + patch;
  }
}
