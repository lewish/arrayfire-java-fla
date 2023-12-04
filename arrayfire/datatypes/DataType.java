package arrayfire.datatypes;

import arrayfire.containers.NativeArray;

public interface DataType<Container extends NativeArray<?, ?, ?>, SumType extends DataType<? ,?>> {

    int code();

    SumType sumType();

    Container create(int length);
}


