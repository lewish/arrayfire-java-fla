package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.F16;

public class F16 implements DataType<F16.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.F16.code();
    }

    public static class Meta implements DataType.Meta<F16, Short, short[]> {

        @Override
        public ValueLayout.OfShort layout() {
            return ValueLayout.JAVA_SHORT;
        }

        @Override
        public F16 sumType() {
            return F16;
        }

        @Override
        public Short get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Short value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public short[] createHeapArray(int length) {
            return new short[length];
        }
    }
}
