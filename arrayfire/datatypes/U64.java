package arrayfire.datatypes;

import arrayfire.containers.U64Array;

public class U64 implements AfDataType<U64Array, U64> {

    @Override
    public int code() {
        return AfDataTypeEnum.U64.code();
    }

    @Override
    public U64 sumType() {
        return AfDataType.U64;
    }

    @Override
    public U64Array create(int length) {
        return new U64Array(length);
    }
}

