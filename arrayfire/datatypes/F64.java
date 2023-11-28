package arrayfire.datatypes;

import arrayfire.containers.F64Array;

public class F64 implements AfDataType<F64Array, arrayfire.datatypes.F64> {

    @Override
    public int code() {
        return AfDataTypeEnum.F64.code();
    }

    @Override
    public arrayfire.datatypes.F64 sumType() {
        return AfDataType.F64;
    }

    @Override
    public F64Array create(int length) {
        return new F64Array(length);
    }
}

