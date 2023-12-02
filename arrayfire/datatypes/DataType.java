package arrayfire.datatypes;

import arrayfire.containers.NativeArray;

public interface DataType<Container extends NativeArray<?, ?, ?>, SumType extends DataType<? ,?>> {

    U64 U64 = new U64();
    U32 U32 = new U32();
    F32 F32 = new F32();
    F16 F16 = new F16();
    F64 F64 = new F64();
    B8 B8 = new B8();
    S32 S32 = new S32();

    int code();

    SumType sumType();

    Container create(int length);
}


