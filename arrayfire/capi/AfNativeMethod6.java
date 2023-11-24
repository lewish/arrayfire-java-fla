package arrayfire.capi;

public class AfNativeMethod6<A, B, C, D, E, F> extends AfNativeMethod {

  public AfNativeMethod6(String name, Class<A> a, Class<B> b, Class<C> c, Class<D> d, Class<E> e, Class<F> f) {
    super(name, a, b, c, d, e, f);
  }

  public void invoke(A a, B b, C c, D d, E e, F f) {
    try {
      handleArrayFireResponse(handle.invoke(a, b, c, d, e, f));
    } catch (Throwable err) {
      if (err instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(err);
    }
  }
}
