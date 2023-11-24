package arrayfire.capi;

public class AfNativeMethod3<A, B, C> extends AfNativeMethod {

  public AfNativeMethod3(String name, Class<A> a, Class<B> b, Class<C> c) {
    super(name, a, b, c);
  }

  public void invoke(A a, B b, C c) {
    try {
      handleArrayFireResponse(handle.invoke(a, b, c));
    } catch (Throwable e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }
}
