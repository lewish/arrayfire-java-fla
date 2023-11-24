package fade.context;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.concurrent.atomic.AtomicBoolean;

@RunWith(JUnit4.class)
public class ContextTest {

  private final Contextual<Integer> testContextual = Contextual.named("test", 1);

  @Test
  public void basic() {
    Assert.assertEquals((Integer) 1, testContextual.get());
    var wasCalled = new AtomicBoolean();
    Context.fork(testContextual.create(2), () -> {
      Assert.assertEquals((Integer) 2, testContextual.get());
      Context.fork(testContextual.create(3), () -> {
        Assert.assertEquals((Integer) 3, testContextual.get());
      });
      Assert.assertEquals((Integer) 2, testContextual.get());
      wasCalled.set(true);
    });
    Assert.assertTrue(wasCalled.get());
  }
}
