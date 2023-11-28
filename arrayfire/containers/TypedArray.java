package arrayfire.containers;

import arrayfire.MemoryContainer;
import arrayfire.Scope;
import arrayfire.datatypes.AfDataType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.af;

public abstract class TypedArray<DataType extends AfDataType<?, ?>, JavaType, JavaArrayType> implements MemoryContainer {

    final DataType type;
    final int length;
    final Arena arena;
    final MemorySegment segment;

    TypedArray(DataType type, int length) {
        this.type = type;
        this.length = length;
        this.arena = Arena.ofShared();
        this.segment = arena.allocateArray(layout(), length);
        af.currentScope().track(this);
    }

    public Arena arena() {
        return arena;
    }

    public MemorySegment segment() {
        return segment;
    }

    public int length() {
        return length;
    }

    public int code() {
        return type.code();
    }

    public DataType type() {
        return type;
    }

    abstract ValueLayout layout();

    abstract JavaType get(int index);

    public abstract void set(int index, JavaType value);

    abstract JavaArrayType toHeap();

    @Override
    public void dispose() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }
}
