package arrayfire.datatypes;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class U32 implements AfDataType<Integer> {

  @Override
  public ValueLayout.OfInt layout() {
    return ValueLayout.JAVA_INT;
  }

  @Override
  public int code() {
    return AfDataTypeEnum.U32.code();
  }

  @Override
  public Integer get(MemorySegment segment, int index) {
    return segment.getAtIndex(layout(), index);
  }

  @Override
  public void set(MemorySegment segment, int index, Integer value) {
    segment.setAtIndex(layout(), index, value);
  }

  public Accessor<Integer> accessor(MemorySegment segment) {
    return new Accessor<>(this, segment);
  }
}
