package arrayfire.datatypes;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static arrayfire.ArrayFire.af;

@RunWith(JUnit4.class)
public class AfDataTypeTest {

  @Test
  public void decodeEncodeF16() {
    var values = new Float[] { 0.0f, 1.0f, -1f, 100.0f, -50f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
    af.scope(() -> {
      var array = af.hostTensor(AfDataType.F16, values);
      var decodedValues = new Float[values.length];
      for (int i = 0; i < values.length; i++) {
        decodedValues[i] = array.get(AfDataType.F16, i);
      }
      Assert.assertArrayEquals(values, decodedValues);
    });
  }
}
