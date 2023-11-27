package arrayfire;

import arrayfire.datatypes.AfDataType;
import arrayfire.numbers.N;

import java.lang.foreign.MemorySegment;

public class RawTensor<DT extends AfDataType<?>> extends Tensor<DT, N, N, N, N> {
    RawTensor(Scope scope, MemorySegment segment, DT type, Shape<?, ?, ?, ?> shape) {
        super(scope, segment, type, ArrayFire.af.shape(ArrayFire.af.n(shape.d0().intValue()), ArrayFire.af.n(shape.d0().intValue()), ArrayFire.af.n(shape.d0().intValue()), ArrayFire.af.n(shape.d0().intValue())));
    }
}
