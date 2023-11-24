package arrayfire.capi;

public class AfNativeMethodVoid2<A, B> extends AfNativeMethod {

  public AfNativeMethodVoid2(String name, Class<A> a, Class<B> b) {
    super(name, a, b);
  }

  public void invoke(A a, B b) {
    try {
      handleArrayFireResponse(handle.invoke(a, b));
    } catch (Throwable ex) {
      throw new RuntimeException(ex);
    }
  }
}
