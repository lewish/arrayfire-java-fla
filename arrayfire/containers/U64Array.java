package arrayfire.containers;

import arrayfire.U64;

import java.lang.foreign.ValueLayout;

import static arrayfire.af.U64;

public class U64Array extends NativeArray<U64, Long, long[]> {

    public U64Array(int length) {
        super(U64, length);
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
