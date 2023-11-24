package arrayfire.capi;

public class AfNativeMethod9<A, B, C, D, E, F, G, H, I> extends AfNativeMethod {

  public AfNativeMethod9(String name,
                         Class<A> a,
                         Class<B> b,
                         Class<C> c,
                         Class<D> d,
                         Class<E> e,
                         Class<F> f,
                         Class<G> g,
                         Class<H> h,
                         Class<I> i) {
    super(name, a, b, c, d, e, f, g, h, i);
  }

  public void invoke(A a, B b, C c, D d, E e, F f, G g, H h, I i) {
    try {
      handleArrayFireResponse(handle.invoke(a, b, c, d, e, f, g, h, i));
    } catch (Throwable err) {
      if (err instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(err);
    }
  }
}
