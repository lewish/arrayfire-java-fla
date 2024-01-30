package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.N;

import java.util.Arrays;
import java.util.function.Function;


public record Shape<D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>>(D0 d0, D1 d1, D2 d2, D3 d3) {

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
}
