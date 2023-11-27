package arrayfire;

import arrayfire.datatypes.AfDataType;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static arrayfire.ArrayFire.af;

@RunWith(JUnit4.class)
public class NewTest {

    @Test
    public void add() {
        var a = af.create(0.0f, 1.0f);
        var b = af.create(1.0f, 2.0f);
        var c = af.add(a, b);
        var data = af.data(c);
        Assert.assertArrayEquals(new float[]{1.0f, 3.0f}, data.toHeap(), 0.0f);
    }

    @Test
    public void sumB8() {
        var a = af.create(AfDataType.B8, true, false, true, true).reshape(2, 2);
        var sum = af.sum(a);
        var data = af.data(sum);
        Assert.assertArrayEquals(new int[]{1, 2}, data.toHeap());
    }
}
