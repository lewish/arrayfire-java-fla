package arrayfire.containers;

import arrayfire.datatypes.AfDataType;
import arrayfire.datatypes.AfDataTypeEnum;
import arrayfire.datatypes.F32;

import java.lang.foreign.ValueLayout;

public class F32Array extends TypedArray<F32, Float, float[]> {

    public F32Array(int length) {
        super(AfDataType.F32, length);
    }

    @Override
    public ValueLayout.OfFloat layout() {
        return ValueLayout.JAVA_FLOAT;
    }

    @Override
    public Float get(int index) {
        return segment.getAtIndex(layout(), index);
    }

    @Override
    public void set(int index, Float value) {
        segment.setAtIndex(layout(), index, value);
    }

    @Override
    public float[] toHeap() {
        var array = new float[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }

    public static F32Array fromHeap(float[] array) {
        var f32Array = new F32Array(array.length);
        for (int i = 0; i < array.length; i++) {
            f32Array.set(i, array[i]);
        }
        return f32Array;
    }
}
