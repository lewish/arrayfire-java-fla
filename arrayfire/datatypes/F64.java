package arrayfire.datatypes;

import arrayfire.containers.F64Array;

public class F64 implements DataType<F64Array, F64> {

    @Override
    public int code() {
        return DataTypeEnum.F64.code();
    }

    @Override
    public arrayfire.datatypes.F64 sumType() {
        return DataType.F64;
    }

    @Override
    public F64Array create(int length) {
        return new F64Array(length);
    }
}

