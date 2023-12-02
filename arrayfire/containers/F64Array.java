package arrayfire.containers;

import arrayfire.datatypes.DataType;
import arrayfire.datatypes.F64;

import java.lang.foreign.ValueLayout;

public class F64Array extends NativeArray<F64, Double, double[]> {

    public F64Array(int length) {
        super(DataType.F64, length);
    }

    @Override
    public ValueLayout.OfDouble layout() {
        return ValueLayout.JAVA_DOUBLE;
    }

    @Override
    public Double get(int index) {
        return segment.getAtIndex(layout(), index);
    }

    @Override
    public void set(int index, Double value) {
        segment.setAtIndex(layout(), index, value);
    }

    @Override
    public double[] toHeap() {
        var array = new double[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
