package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.S64;

public class S64 implements DataType<S64.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.S64.code();
    }

    public static class Meta implements DataType.Meta<S64, Long, long[]> {

        @Override
        public ValueLayout.OfLong layout() {
            return ValueLayout.JAVA_LONG;
        }

        @Override
        public S64 sumType() {
            return S64;
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

