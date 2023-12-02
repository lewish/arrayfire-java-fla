package arrayfire.containers;

import arrayfire.datatypes.DataType;
import arrayfire.datatypes.U32;

import java.lang.foreign.ValueLayout;

public class U32Array extends NativeArray<U32, Integer, int[]> {

    public U32Array(int length) {
        super(DataType.U32, length);
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
    public int[] toHeap() {
        var array = new int[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
