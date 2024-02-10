package arrayfire;

import java.lang.foreign.MemorySegment;

public class ErrorArray<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> extends Array<T, S> {
    private final String errorMessage;

    public ErrorArray(T type, S shape, String errorMessage) {
        super(type, shape);
        this.errorMessage = errorMessage;
    }

    @Override
    public MemorySegment dereference() {
        throw new UnsupportedOperationException(errorMessage);
    }

    @Override
    public MemorySegment segment() {
        throw new UnsupportedOperationException(errorMessage);
    }
}
