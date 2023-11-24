package arrayfire.datatypes;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class F16 implements AfDataType<Float> {

  private static short encode(float value) {
    int fltInt32 = Float.floatToRawIntBits(value);
    int fltInt16;

    fltInt16 = (fltInt32 >>> 31) << 5;
    int tmp = (fltInt32 >>> 23) & 0xff;
    tmp = (tmp - 0x70) & ((0x70 - tmp) >> 4 >>> 27);
    fltInt16 = (fltInt16 | tmp) << 10;
    fltInt16 |= (fltInt32 >> 13) & 0x3ff;
    return (short) fltInt16;
  }

  @Override
  public MemoryLayout layout() {
    return ValueLayout.JAVA_SHORT;
  }

  @Override
  public int code() {
    return AfDataTypeEnum.F16.code();
  }

  @Override
  public Float get(MemorySegment segment, int index) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void set(MemorySegment segment, int index, Float value) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Accessor<Float> accessor(MemorySegment segment) {
    return new Accessor<>(this, segment);
  }
}
