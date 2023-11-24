package arrayfire.datatypes;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class F32 implements AfDataType<Float> {

  @Override
  public ValueLayout.OfFloat layout() {
    return ValueLayout.JAVA_FLOAT;
  }

  @Override
  public int code() {
    return AfDataTypeEnum.F32.code();
  }

  @Override
  public Float get(MemorySegment segment, int index) {
    return segment.getAtIndex(layout(), index);
  }

  @Override
  public void set(MemorySegment segment, int index, Float value) {
    segment.setAtIndex(layout(), index, value);
  }

  @Override
  public Accessor<Float> accessor(MemorySegment segment) {
    return new Accessor<>(this, segment);
  }
}
