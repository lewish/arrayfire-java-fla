package arrayfire.datatypes;

import arrayfire.containers.S32Array;

import static arrayfire.af.S32;

public class S32 implements DataType<S32Array, S32> {

    @Override
    public int code() {
        return DataTypeEnum.S32.code();
    }

    @Override
    public S32 sumType() {
        return S32;
    }

    @Override
    public S32Array create(int length) {
        return new S32Array(length);
    }
}

