package arrayfire.containers;

import arrayfire.S64;

import java.lang.foreign.ValueLayout;

import static arrayfire.af.S64;

public class S64Array extends NativeArray<S64, Long, long[]> {

    public S64Array(int length) {
        super(S64, length);
    }

    @Override
    public ValueLayout.OfLong layout() {
        return ValueLayout.JAVA_LONG;
    }

    @Override
    public Long get(int index) {
        return segment.getAtIndex(layout(), index);
    }

    @Override
    public void set(int index, Long value) {
        segment.setAtIndex(layout(), index, value);
    }

    @Override
    public long[] java() {
        var array = new long[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
