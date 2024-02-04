package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.U32;

public class U32 implements DataType<U32.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.U32.code();
    }

    public static class Meta implements DataType.Meta<U32, Integer, int[]> {

        @Override
        public ValueLayout.OfInt layout() {
            return ValueLayout.JAVA_INT;
        }

        @Override
        public U32 sumType() {
            return U32;
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

