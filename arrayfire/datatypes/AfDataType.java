package arrayfire.datatypes;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

public interface AfDataType<JT> {

  U64 U64 = new U64();
  U32 U32 = new U32();
  F32 F32 = new F32();
  F16 F16 = new F16();
  F64 F64 = new F64();
  B8 B8 = new B8();

  MemoryLayout layout();

  int code();

  JT get(MemorySegment segment, int index);

  void set(MemorySegment segment, int index, JT value);

  Accessor<JT> accessor(MemorySegment segment);

  record Accessor<JT>(AfDataType<JT> dt, MemorySegment segment) {

    public JT get(int index) {
      return dt.get(segment, index);
    }

    public void set(int index, JT value) {
      dt.set(segment, index, value);
    }
  }
}


