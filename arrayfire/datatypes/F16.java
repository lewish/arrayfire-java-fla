package arrayfire.datatypes;

import arrayfire.containers.F16Array;

public class F16 implements AfDataType<F16Array, F16> {

  @Override
  public int code() {
    return AfDataTypeEnum.F16.code();
  }

  @Override
  public F16 sumType() {
    return AfDataType.F16;
  }

  @Override
  public F16Array create(int length) {
    return new F16Array(length);
  }

}
