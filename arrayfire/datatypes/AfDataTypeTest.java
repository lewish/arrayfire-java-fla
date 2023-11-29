package arrayfire.datatypes;

import arrayfire.containers.F16Array;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static arrayfire.ArrayFire.af;

@RunWith(JUnit4.class)
public class AfDataTypeTest {

  @Test
  public void decodeEncodeF16() {
    var values = new float[] { 0.0f, 1.0f, -1f, 100.0f, -50f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
    af.tidy(() -> {
      var result = af.data(af.create(AfDataType.F16, values)).toHeap();
      Assert.assertArrayEquals(values, result, 1E-5f);
    });
  }
}
