package arrayfire.containers;

import arrayfire.datatypes.DataType;
import arrayfire.datatypes.B8;

import java.lang.foreign.ValueLayout;

public class B8Array extends NativeArray<B8, Boolean, boolean[]> {

    public B8Array(int length) {
        super(DataType.B8, length);
    }

    @Override
    public ValueLayout.OfBoolean layout() {
        return ValueLayout.JAVA_BOOLEAN;
    }

    @Override
    public Boolean get(int index) {
        return segment.getAtIndex(layout(), index);
    }

    @Override
    public void set(int index, Boolean value) {
        segment.setAtIndex(layout(), index, value);
    }

    @Override
    public boolean[] toHeap() {
        var array = new boolean[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
