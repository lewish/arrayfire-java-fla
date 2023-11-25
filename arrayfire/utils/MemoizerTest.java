package arrayfire.utils;

import arrayfire.utils.Memoizer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.function.Supplier;

@RunWith(JUnit4.class)
public class MemoizerTest {

  @Test
  public void testMemoizer() {
    int[] counter = {0};
    Supplier<Integer> supplier = () -> {
      counter[0]++;
      return counter[0];
    };
    Supplier<Integer> memoize = Memoizer.memoize(supplier);
    assert memoize.get() == 1;
  }
}
