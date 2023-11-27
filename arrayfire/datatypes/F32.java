package arrayfire.datatypes;

import arrayfire.containers.F16Array;
import arrayfire.containers.F32Array;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class F32 implements AfDataType<F32Array, F32> {

  @Override
  public int code() {
    return AfDataTypeEnum.F32.code();
  }

  @Override
  public arrayfire.datatypes.F32 sumType() {
    return AfDataType.F32;
  }

  @Override
  public F32Array create(int length) {
    return new F32Array(length);
  }
}

