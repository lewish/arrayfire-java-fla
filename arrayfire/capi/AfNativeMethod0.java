package arrayfire.capi;

public class AfNativeMethod0 extends AfNativeMethod {

  public AfNativeMethod0(String name) {
    super(name);
  }

  public void invoke() {
    try {
      handleArrayFireResponse(handle.invoke());
    } catch (Throwable e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }
}
