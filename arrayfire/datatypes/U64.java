package arrayfire.datatypes;

import arrayfire.containers.U64Array;

import static arrayfire.af.U64;

public class U64 implements DataType<U64Array, U64> {

    @Override
    public int code() {
        return DataTypeEnum.U64.code();
    }

    @Override
    public U64 sumType() {
        return U64;
    }

    @Override
    public U64Array create(int length) {
        return new U64Array(length);
    }
}

