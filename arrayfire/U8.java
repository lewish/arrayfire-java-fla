package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.U32;

public class U8 implements DataType<U8.Meta> {

    public static final Meta META = new Meta();

    @Override
    public Meta meta() {
        return META;
    }

    @Override
    public int code() {
        return DataTypeEnum.U8.code();
    }

    public static class Meta implements DataType.Meta<U32, Byte, byte[]> {

        @Override
        public ValueLayout.OfByte layout() {
            return ValueLayout.JAVA_BYTE;
        }

        @Override
        public U32 sumType() {
            return U32;
        }

        @Override
        public Byte get(MemorySegment segment, int index) {
            return segment.getAtIndex(layout(), index);
        }

        @Override
        public void set(MemorySegment segment, int index, Byte value) {
            segment.setAtIndex(layout(), index, value);
        }

        @Override
        public byte[] createHeapArray(int length) {
            return new byte[length];
        }
    }
}

