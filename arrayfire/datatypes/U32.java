package arrayfire.datatypes;

import arrayfire.containers.U32Array;

public class U32 implements AfDataType<U32Array, U32> {

    @Override
    public int code() {
        return AfDataTypeEnum.F32.code();
    }

    @Override
    public U32 sumType() {
        return AfDataType.U32;
    }

    @Override
    public U32Array create(int length) {
        return new U32Array(length);
    }
}

