package arrayfire.capi;

public class AfNativeMethod7<A, B, C, D, E, F, G> extends AfNativeMethod {

  public AfNativeMethod7(String name,
                         Class<A> a,
                         Class<B> b,
                         Class<C> c,
                         Class<D> d,
                         Class<E> e,
                         Class<F> f,
                         Class<G> g) {
    super(name, a, b, c, d, e, f, g);
  }

  public void invoke(A a, B b, C c, D d, E e, F f, G g) {
    try {
      handleArrayFireResponse(handle.invoke(a, b, c, d, e, f, g));
    } catch (Throwable err) {
      if (err instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(err);
    }
  }
}
