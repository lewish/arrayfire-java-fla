package arrayfire;

import arrayfire.numbers.Num;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.Function;

/*
 * https://arrayfire.org/docs/index_8h_source.htm
 */
public class Index<D extends Num<D>> {
    static MemoryLayout LAYOUT = MemoryLayout.structLayout(
        MemoryLayout.unionLayout(ValueLayout.ADDRESS.withName("arr"), Seq.LAYOUT.withName("seq")).withName("union"),
        ValueLayout.JAVA_BOOLEAN.withName("isSeq"), ValueLayout.JAVA_BOOLEAN.withName("isBatch"),
        MemoryLayout.paddingLayout(6));

    private final Array<?, ?> arr;
    private final Seq seq;

    private final Function<Integer, D> generator;

    Index(Array<?, ?> arr, Function<Integer, D> generator) {
        this.arr = arr;
        this.seq = null;
        this.generator = generator;
    }

    Index(Seq seq, Function<Integer, D> generator) {
        this.arr = null;
        this.seq = seq;
        this.generator = generator;
    }

    public D createDim() {
        return generator.apply(size());
    }

    public D createDim(int size) {
        return generator.apply(size);
    }

    void emigrate(MemorySegment segment) {
        if (arr != null) {
            segment.set(ValueLayout.ADDRESS,
                LAYOUT.byteOffset(PathElement.groupElement("union"), PathElement.groupElement("arr")),
                arr.dereference());
        }
        if (seq != null) {
            seq.emigrate(
                segment.asSlice(LAYOUT.byteOffset(PathElement.groupElement("union"), PathElement.groupElement("seq"))));
        }
        segment.set(ValueLayout.JAVA_BOOLEAN, LAYOUT.byteOffset(PathElement.groupElement("isSeq")), seq != null);
        segment.set(ValueLayout.JAVA_BOOLEAN, LAYOUT.byteOffset(PathElement.groupElement("isBatch")), false);
    }

    int size() {
        if (seq != null) {
            return seq.size();
        }
        if (arr != null) {
            return arr.capacity();
        }
        throw new RuntimeException();
    }
}
