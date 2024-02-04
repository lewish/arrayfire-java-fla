package arrayfire;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public interface DataType<DTM extends DataType.Meta<?, ?, ?>> {

    DTM meta();

    int code();

    interface Meta<SumType extends DataType<?>, JavaType, JavaArrayType> {

        public SumType sumType();

        public ValueLayout layout();

        public JavaType get(MemorySegment segment, int index);

        public void set(MemorySegment segment, int index, JavaType value);

        public JavaArrayType createHeapArray(int length);
    }
}


