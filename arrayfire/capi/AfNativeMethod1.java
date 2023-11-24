package arrayfire.capi;

public class AfNativeMethod1<A> extends AfNativeMethod {

  public AfNativeMethod1(String name, Class<A> a) {
    super(name, a);
  }

  public void invoke(A a) {
    try {
      handleArrayFireResponse(handle.invoke(a));
    } catch (Throwable e) {
      if (e instanceof RuntimeException re) {
        throw re;
      }
      throw new RuntimeException(e);
    }
  }
}
