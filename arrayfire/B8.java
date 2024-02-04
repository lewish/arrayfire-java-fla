package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.U32;

public class B8 implements DataType<B8.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.B8.code();
    }

    public static class Meta implements DataType.Meta<U32, Boolean, boolean[]> {

        @Override
        public ValueLayout.OfBoolean layout() {
            return ValueLayout.JAVA_BOOLEAN;
        }

        @Override
        public U32 sumType() {
            return U32;
        }

        @Override
        public Boolean get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Boolean value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public boolean[] createHeapArray(int length) {
            return new boolean[length];
        }
    }
}

