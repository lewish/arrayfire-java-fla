package arrayfire.datatypes;

import arrayfire.containers.B8Array;
import arrayfire.containers.U8Array;

import static arrayfire.af.U32;

public class U8 implements DataType<U8Array, U32> {

    @Override
    public int code() {
        return DataTypeEnum.U8.code();
    }

    @Override
    public U32 sumType() {
        return U32;
    }

    @Override
    public U8Array create(int length) {
        return new U8Array(length);
    }
}

