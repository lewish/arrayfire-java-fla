package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.F32;

public class F32 implements DataType<F32.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.F32.code();
    }

    public static class Meta implements DataType.Meta<F32, Float, float[]> {

        @Override
        public ValueLayout.OfFloat layout() {
            return ValueLayout.JAVA_FLOAT;
        }

        @Override
        public F32 sumType() {
            return F32;
        }

        @Override
        public Float get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Float value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public float[] createHeapArray(int length) {
            return new float[length];
        }
    }
}

