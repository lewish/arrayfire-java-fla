package arrayfire.datatypes;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class U64 implements AfDataType<Long> {

  @Override
  public ValueLayout.OfLong layout() {
    return ValueLayout.JAVA_LONG;
  }

  @Override
  public int code() {
    return AfDataTypeEnum.U64.code();
  }

  @Override
  public Long get(MemorySegment segment, int index) {
    return segment.getAtIndex(layout(), index);
  }

  @Override
  public void set(MemorySegment segment, int index, Long value) {
    segment.setAtIndex(layout(), index, value);
  }

  public Accessor<Long> accessor(MemorySegment segment) {
    return new Accessor<>(this, segment);
  }
}
