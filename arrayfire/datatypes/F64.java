package arrayfire.datatypes;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class F64 implements AfDataType<Double> {

  @Override
  public ValueLayout.OfDouble layout() {
    return ValueLayout.JAVA_DOUBLE;
  }

  @Override
  public int code() {
    return AfDataTypeEnum.F64.code();
  }

  @Override
  public Double get(MemorySegment segment, int index) {
    return segment.getAtIndex(layout(), index);
  }

  @Override
  public void set(MemorySegment segment, int index, Double value) {
    segment.setAtIndex(layout(), index, value);
  }

  @Override
  public Accessor<Double> accessor(MemorySegment segment) {
    return new Accessor<>(this, segment);
  }
}
