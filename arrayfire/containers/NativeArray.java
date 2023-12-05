package arrayfire.containers;

import arrayfire.MemoryContainer;
import arrayfire.af;
import arrayfire.datatypes.DataType;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * @param <DT>  Data type
 * @param <JT>  Java type
 * @param <JAT> Java array type
 */
public abstract class NativeArray<DT extends DataType<?, ?>, JT, JAT> implements MemoryContainer {

    final DT type;
    final int length;
    final Arena arena;
    final MemorySegment segment;

    NativeArray(DT type, int length) {
        this.type = type;
        this.length = length;
        this.arena = Arena.ofShared();
        this.segment = arena.allocateArray(layout(), length);
        af.memoryScope().track(this);
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

    public DT type() {
        return type;
    }

    abstract ValueLayout layout();

    abstract JT get(int index);

    public abstract void set(int index, JT value);

    abstract JAT java();

    @Override
    public void dispose() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }
}
