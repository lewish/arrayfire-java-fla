package arrayfire.capi;

public class AfNativeMethod5<A, B, C, D, E> extends AfNativeMethod {

  public AfNativeMethod5(String name, Class<A> a, Class<B> b, Class<C> c, Class<D> d, Class<E> e) {
    super(name, a, b, c, d, e);
  }

  public void invoke(A a, B b, C c, D d, E e) {
    try {
      handleArrayFireResponse(handle.invoke(a, b, c, d, e));
    } catch (Throwable err) {
      if (err instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(err);
    }
  }
}
