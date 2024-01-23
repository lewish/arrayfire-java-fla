package arrayfire;

import arrayfire.containers.F64Array;

import static arrayfire.af.F64;

public class F64 implements DataType<F64Array, F64> {

    @Override
    public int code() {
        return DataTypeEnum.F64.code();
    }

    @Override
    public arrayfire.F64 sumType() {
        return F64;
    }

    @Override
    public F64Array create(int length) {
        return new F64Array(length);
    }
}

