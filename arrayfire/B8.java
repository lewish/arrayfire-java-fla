package arrayfire;

import arrayfire.containers.B8Array;

import static arrayfire.af.U32;

public class B8 implements DataType<B8Array, U32> {

    @Override
    public int code() {
        return DataTypeEnum.B8.code();
    }

    @Override
    public U32 sumType() {
        return U32;
    }

    @Override
    public B8Array create(int length) {
        return new B8Array(length);
    }
}

