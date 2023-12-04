package arrayfire.datatypes;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static arrayfire.ArrayFire.af;

@RunWith(JUnit4.class)
public class DataTypeTest {

    @Test
    public void decodeEncodeF16() {
        var values = new float[]{0.0f, 1.0f, -1f, 100.0f, -50f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
        af.tidy(() -> {
            var result = af.data(af.create(DataType.F16, values)).toHeap();
            Assert.assertArrayEquals(values, result, 1E-5f);
        });
    }

    @Test
    public void decodeEncodeU32() {
        var values = new int[]{0, 1, 2, Integer.MAX_VALUE};
        af.tidy(() -> {
            var result = af.data(af.create(DataType.U32, values)).toHeap();
            Assert.assertArrayEquals(values, result);
        });
    }
}
