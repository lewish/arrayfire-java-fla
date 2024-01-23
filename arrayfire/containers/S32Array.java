package arrayfire.containers;

import arrayfire.S32;

import java.lang.foreign.ValueLayout;

import static arrayfire.af.S32;

public class S32Array extends NativeArray<S32, Integer, int[]> {

    public S32Array(int length) {
        super(S32, length);
    }

    @Override
    public ValueLayout.OfInt layout() {
        return ValueLayout.JAVA_INT;
    }

    @Override
    public Integer get(int index) {
        return segment.getAtIndex(layout(), index);
    }

    @Override
    public void set(int index, Integer value) {
        segment.setAtIndex(layout(), index, value);
    }

    @Override
    public int[] java() {
        var array = new int[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
