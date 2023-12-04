package arrayfire.datatypes;

import arrayfire.containers.F32Array;

import static arrayfire.af.F32;

public class F32 implements DataType<F32Array, F32> {

    @Override
    public int code() {
        return DataTypeEnum.F32.code();
    }

    @Override
    public arrayfire.datatypes.F32 sumType() {
        return F32;
    }

    @Override
    public F32Array create(int length) {
        return new F32Array(length);
    }
}

