package arrayfire.containers;

import arrayfire.datatypes.AfDataType;
import arrayfire.datatypes.U64;

import java.lang.foreign.ValueLayout;

public class U64Array extends TypedArray<U64, Long, long[]> {

    public U64Array(int length) {
        super(AfDataType.U64, length);
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
    public long[] toHeap() {
        var array = new long[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }

    public static U64Array fromHeap(long[] array) {
        var u32Array = new U64Array(array.length);
        for (int i = 0; i < array.length; i++) {
            u32Array.set(i, array[i]);
        }
        return u32Array;
    }
}
