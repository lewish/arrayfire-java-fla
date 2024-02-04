package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.N;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.Function;


public class Shape<D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> {
    private final D0 d0;
    private final D1 d1;
    private final D2 d2;
    private final D3 d3;

    public Shape(D0 d0, D1 d1, D2 d2, D3 d3) {
        this.d0 = d0;
        this.d1 = d1;
        this.d2 = d2;
        this.d3 = d3;
    }

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

    public D0 d0() {
        return d0;
    }

    public D1 d1() {
        return d1;
    }

    public D2 d2() {
        return d2;
    }

    public D3 d3() {
        return d3;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this)
            return true;
        if (obj == null || obj.getClass() != this.getClass())
            return false;
        var that = (Shape<? ,? ,? ,?>) obj;
        return Objects.equals(this.d0, that.d0) && Objects.equals(this.d1, that.d1) && Objects.equals(this.d2, that.d2) &&
                   Objects.equals(this.d3, that.d3);
    }

    @Override
    public int hashCode() {
        return Objects.hash(d0, d1, d2, d3);
    }

}
