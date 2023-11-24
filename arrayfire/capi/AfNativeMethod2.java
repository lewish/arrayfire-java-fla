package arrayfire.capi;

public class AfNativeMethod2<A, B> extends AfNativeMethod {

  public AfNativeMethod2(String name, Class<A> a, Class<B> b) {
    super(name, a, b);
  }

  public void invoke(A a, B b) {
    try {
      handleArrayFireResponse(handle.invoke(a, b));
    } catch (Throwable e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }
}
