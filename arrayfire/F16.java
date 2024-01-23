package arrayfire;

import arrayfire.containers.F16Array;
import static arrayfire.af.F16;

public class F16 implements DataType<F16Array, F16> {

  @Override
  public int code() {
    return DataTypeEnum.F16.code();
  }

  @Override
  public F16 sumType() {
    return F16;
  }

  @Override
  public F16Array create(int length) {
    return new F16Array(length);
  }

}
