package arrayfire.datatypes;

import arrayfire.af;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DataTypeTest {

    @Test
    public void decodeEncodeF16() {
        var values = new float[]{0.0f, 1.0f, -1f, 100.0f, -50f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
        af.tidy(() -> {
            var result = af.data(af.create(af.F16, values)).java();
            Assert.assertArrayEquals(values, result, 1E-5f);
        });
    }

    @Test
    public void decodeEncodeU32() {
        var values = new int[]{0, 1, 2, Integer.MAX_VALUE};
        af.tidy(() -> {
            var result = af.data(af.create(af.U32, values)).java();
            Assert.assertArrayEquals(values, result);
        });
    }
}
