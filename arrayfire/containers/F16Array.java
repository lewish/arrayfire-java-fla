package arrayfire.containers;

import arrayfire.datatypes.AfDataType;
import arrayfire.datatypes.AfDataTypeEnum;
import arrayfire.datatypes.F16;

import java.lang.foreign.ValueLayout;

public class F16Array extends TypedArray<F16, Float, float[]> {

    public F16Array(int length) {
        super(AfDataType.F16, length);
    }
    @Override
    public ValueLayout.OfShort layout() {
        return ValueLayout.JAVA_SHORT;
    }

    @Override
    public Float get(int index) {
        return Float.float16ToFloat(segment.getAtIndex(layout(), index));
    }

    @Override
    public void set(int index, Float value) {
        segment.setAtIndex(layout(), index, Float.floatToFloat16(value));
    }

    @Override
    public float[] toHeap() {
        var array = new float[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
