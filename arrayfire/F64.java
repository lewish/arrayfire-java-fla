package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.af.F64;

public class F64 implements DataType<F64.Meta> {

    public static final Meta META = new Meta();

    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.F64.code();
    }

    public static class Meta implements DataType.Meta<F64, Double, double[]> {

        @Override
        public ValueLayout.OfDouble layout() {
            return ValueLayout.JAVA_DOUBLE;
        }

        @Override
        public F64 sumType() {
            return F64;
        }

        @Override
        public Double get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Double value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public double[] createHeapArray(int length) {
            return new double[length];
        }
    }
}

