package arrayfire.containers;

import arrayfire.MemoryContainer;
import arrayfire.MemoryScope;
import arrayfire.af;
import arrayfire.DataType;

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

    final boolean pinned;
    final Arena arena;
    final MemorySegment segment;

    NativeArray(DT type, int length) {
        this(type, length, false);
    }

    NativeArray(DT type, int length, boolean pinned) {
        this.type = type;
        this.length = length;
        this.pinned = pinned;
        if (pinned) {
            this.arena = null;
            this.segment = af.allocPinned(length * layout().byteSize());
        } else {
            this.arena = Arena.ofShared();
            this.segment = arena.allocateArray(layout(), length);

        }
        MemoryScope.current().register(this);
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

    public abstract JAT java();

    @Override
    public void dispose() {
        if (pinned) {
            af.freePinned(segment);
        } else {
            if (arena.scope().isAlive()) {
                arena.close();
            }
        }
    }
}
