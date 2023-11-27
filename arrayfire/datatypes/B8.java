package arrayfire.datatypes;

import arrayfire.containers.B8Array;

public class B8 implements AfDataType<B8Array, U32> {

    @Override
    public int code() {
        return AfDataTypeEnum.B8.code();
    }

    @Override
    public U32 sumType() {
        return AfDataType.U32;
    }

    @Override
    public B8Array create(int length) {
        return new B8Array(length);
    }
}

