package arrayfire;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public class HostArray<DT extends DataType<? extends DataType.Meta<?, JT, ?>>, JT, S extends Shape<?, ?, ?, ?>> implements MemoryContainer {

    final DT type;
    final S shape;

    final boolean pinned;
    final Arena arena;
    final MemorySegment segment;

    public HostArray(DT type, S shape, boolean pinned) {
        this.type = type;
        this.shape = shape;
        this.pinned = pinned;
        if (pinned) {
            this.arena = null;
            this.segment = af.allocPinned(shape.capacity() * type.meta().layout().byteSize());
        } else {
            this.arena = Arena.ofShared();
            this.segment = arena.allocateArray(type.meta().layout(), shape.capacity());

        }
        Scope.current().register(this);
    }

    public DT type() {
        return type;
    }

    public S shape() {
        return shape;
    }

    public MemorySegment segment() {
        return segment;
    }

    public int length() {
        return shape.capacity();
    }

    public JT get(int index) {
        return type.meta().get(segment, index);
    }

    public void set(int index, JT value) {
        type.meta().set(segment, index, value);
    }

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

    public String toString() {
        var limit = Math.min(1000, length());
        var sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < limit; i++) {
            sb.append(get(i));
            if (i < limit - 1) {
                sb.append(", ");
            }
        }
        if (limit < length()) {
            sb.append("...");
        }
        sb.append("]");
        return sb.toString();
    }
}
