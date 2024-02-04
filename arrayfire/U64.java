package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.U64;

public class U64 implements DataType<U64.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.U64.code();
    }

    public static class Meta implements DataType.Meta<U64, Long, long[]> {

        @Override
        public ValueLayout.OfLong layout() {
            return ValueLayout.JAVA_LONG;
        }

        @Override
        public U64 sumType() {
            return U64;
        }

        @Override
        public Long get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Long value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public long[] createHeapArray(int length) {
            return new long[length];
        }
    }
}

