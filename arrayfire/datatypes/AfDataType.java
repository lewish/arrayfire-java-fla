package arrayfire.datatypes;

import arrayfire.containers.TypedArray;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

public interface AfDataType<Container extends TypedArray<?, ?, ?>, SumType extends AfDataType<? ,?>> {

    U64 U64 = new U64();
    U32 U32 = new U32();
    F32 F32 = new F32();
    F16 F16 = new F16();
    F64 F64 = new F64();
    B8 B8 = new B8();

    int code();

    SumType sumType();

    Container create(int length);
}


