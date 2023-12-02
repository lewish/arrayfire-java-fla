package arrayfire.datatypes;

import arrayfire.containers.B8Array;

public class B8 implements DataType<B8Array, U32> {

    @Override
    public int code() {
        return DataTypeEnum.B8.code();
    }

    @Override
    public U32 sumType() {
        return DataType.U32;
    }

    @Override
    public B8Array create(int length) {
        return new B8Array(length);
    }
}

