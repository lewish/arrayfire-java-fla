package arrayfire;

import arrayfire.datatypes.U64;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/*
 * https://arrayfire.org/docs/index_8h_source.htm
 */
public class Index {
  static MemoryLayout LAYOUT = MemoryLayout.structLayout(
      MemoryLayout.unionLayout(ValueLayout.ADDRESS.withName("arr"), Seq.LAYOUT.withName("seq")).withName("union"),
      ValueLayout.JAVA_BOOLEAN.withName("isSeq"),
      ValueLayout.JAVA_BOOLEAN.withName("isBatch"),
      MemoryLayout.paddingLayout(6));

  private final Tensor<?, ?, ?, ?, ?> arr;
  private final Seq seq;

  Index(Tensor<?, ?, ?, ?, ?> arr) {
    this.arr = arr;
    this.seq = null;
  }

  Index(Seq seq) {
    this.arr = null;
    this.seq = seq;
  }

  void emigrate(MemorySegment segment) {
    if (arr != null) {
      segment.set(ValueLayout.ADDRESS,
                  LAYOUT.byteOffset(PathElement.groupElement("union"), PathElement.groupElement("arr")),
                  arr.dereference());
    }
    if (seq != null) {
      seq.emigrate(segment.asSlice(LAYOUT.byteOffset(PathElement.groupElement("union"),
                                                     PathElement.groupElement("seq"))));
    }
    segment.set(ValueLayout.JAVA_BOOLEAN, LAYOUT.byteOffset(PathElement.groupElement("isSeq")), seq != null);
    // TODO: Work out what this is for.
    segment.set(ValueLayout.JAVA_BOOLEAN, LAYOUT.byteOffset(PathElement.groupElement("isBatch")), true);
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
