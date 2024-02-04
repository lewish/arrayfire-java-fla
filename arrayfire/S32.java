package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.S32;

public class S32 implements DataType<S32.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.S32.code();
    }

    public static class Meta implements DataType.Meta<S32, Integer, int[]> {

        @Override
        public ValueLayout.OfInt layout() {
            return ValueLayout.JAVA_INT;
        }

        @Override
        public S32 sumType() {
            return S32;
        }

        @Override
        public Integer get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Integer value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public int[] createHeapArray(int length) {
            return new int[length];
        }
    }
}

