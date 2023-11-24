package arrayfire.capi;

public class AfNativeMethod4<A, B, C, D> extends AfNativeMethod {

  public AfNativeMethod4(String name, Class<A> a, Class<B> b, Class<C> c, Class<D> d) {
    super(name, a, b, c, d);
  }

  public void invoke(A a, B b, C c, D d) {
    try {
      handleArrayFireResponse(handle.invoke(a, b, c, d));
    } catch (Throwable e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }
}
