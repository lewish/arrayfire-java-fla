package arrayfire;

import arrayfire.numbers.IntNumber;
import arrayfire.numbers.N;

import java.util.Arrays;
import java.util.function.Function;


public record Shape<D0 extends IntNumber, D1 extends IntNumber, D2 extends IntNumber, D3 extends IntNumber>(D0 d0, D1 d1, D2 d2, D3 d3) {

    public int capacity() {
        return d0.size() * d1.size() * d2.size() * d3.size();
    }

    public long[] dims() {
        return new long[]{d0.size(), d1.size(), d2.size(), d3.size()};
    }

    @Override
    public String toString() {
        return Arrays.toString(dims());
    }

    public N flat() {
        return af.n(capacity());
    }

    public <D extends IntNumber> D flat(Function<Integer, D> generator) {
        return generator.apply(capacity());
    }
}
