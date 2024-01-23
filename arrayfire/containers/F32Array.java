package arrayfire.containers;

import arrayfire.F32;

import java.lang.foreign.ValueLayout;

import static arrayfire.af.F32;

public class F32Array extends NativeArray<F32, Float, float[]> {

    public F32Array(int length) {
        super(F32, length);
    }

    public F32Array(int length, boolean pinned) {
        super(F32, length, pinned);
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
    public float[] java() {
        var array = new float[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
