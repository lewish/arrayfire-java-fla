package arrayfire;

import arrayfire.numbers.N;

import java.util.Arrays;
import java.util.function.Function;


public record Shape<D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number>(D0 d0, D1 d1, D2 d2,
                                                                                                D3 d3) {

    public int capacity() {
        return d0.intValue() * d1.intValue() * d2.intValue() * d3.intValue();
    }

    public long[] dims() {
        return new long[]{d0.intValue(), d1.intValue(), d2.intValue(), d3.intValue()};
    }

    @Override
    public String toString() {
        return Arrays.toString(dims());
    }

    public N flat() {
        return af.n(capacity());
    }

    public <D extends Number> D flat(Function<Integer, D> generator) {
        return generator.apply(capacity());
    }
}
