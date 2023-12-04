package arrayfire.datatypes;

import arrayfire.containers.U32Array;

public class U32 implements DataType<U32Array, U32> {

    @Override
    public int code() {
        return DataTypeEnum.U32.code();
    }

    @Override
    public U32 sumType() {
        return DataType.U32;
    }

    @Override
    public U32Array create(int length) {
        return new U32Array(length);
    }
}

