package arrayfire.datatypes;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class B8 implements AfDataType<Boolean> {

  @Override
  public ValueLayout.OfBoolean layout() {
    return ValueLayout.JAVA_BOOLEAN;
  }

  @Override
  public int code() {
    return AfDataTypeEnum.B8.code();
  }

  @Override
  public Boolean get(MemorySegment segment, int index) {
    return segment.get(layout(), index);
  }

  @Override
  public void set(MemorySegment segment, int index, Boolean value) {
    segment.set(layout(), index, value);
  }

  @Override
  public Accessor<Boolean> accessor(MemorySegment segment) {
    return new Accessor<>(this, segment);
  }
}
