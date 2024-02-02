package arrayfire;

import arrayfire.containers.U64Array;

import static arrayfire.af.S64;

public class S64 implements DataType<U64Array, S64> {

    @Override
    public int code() {
        return DataTypeEnum.S64.code();
    }

    @Override
    public S64 sumType() {
        return S64;
    }

    @Override
    public U64Array create(int length) {
        return new U64Array(length);
    }
}

