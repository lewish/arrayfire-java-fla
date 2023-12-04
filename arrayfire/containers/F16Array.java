package arrayfire.containers;

import arrayfire.datatypes.F16;

import java.lang.foreign.ValueLayout;

import static arrayfire.af.F16;

public class F16Array extends NativeArray<F16, Float, float[]> {

    public F16Array(int length) {
        super(F16, length);
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
    public float[] java() {
        var array = new float[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
