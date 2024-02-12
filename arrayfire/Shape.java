package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.U;

import java.util.Arrays;
import java.util.Objects;


public class Shape<D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> {
    private final D0 d0;
    private final D1 d1;
    private final D2 d2;
    private final D3 d3;

    private final long[] dims;

    public Shape(D0 d0, D1 d1, D2 d2, D3 d3) {
        this.d0 = d0;
        this.d1 = d1;
        this.d2 = d2;
        this.d3 = d3;
        if (!(d3 instanceof U)) {
            dims = new long[]{d0.size(), d1.size(), d2.size(), d3.size()};
        } else if (!(d2 instanceof U)) {
            dims = new long[]{d0.size(), d1.size(), d2.size()};
        } else if (!(d1 instanceof U)) {
            dims = new long[]{d0.size(), d1.size()};
        } else {
            dims = new long[]{d0.size()};
        }
    }

    public int capacity() {
        return d0.size() * d1.size() * d2.size() * d3.size();
    }

    public int ndims() {
        return dims.length;
    }

    public long[] dims() {
        return dims;
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

    public int offset(int d0i, int d1i, int d2i, int d3i) {
        return d3i * d2.size() * d1.size() * d0.size() + d2i * d1.size() * d0.size() + d1i * d0.size() + d0i;
    }

    public int offset(int d0i, int d1i, int d2i) {
        return d2i * d1.size() * d0.size() + d1i * d0.size() + d0i;
    }

    public int offset(int d0i, int d1i) {
        return d1i * d0.size() + d0i;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this)
            return true;
        if (obj == null || obj.getClass() != this.getClass())
            return false;
        var that = (Shape<?, ?, ?, ?>) obj;
        return Objects.equals(this.d0, that.d0) && Objects.equals(this.d1, that.d1) &&
                   Objects.equals(this.d2, that.d2) && Objects.equals(this.d3, that.d3);
    }

    @Override
    public int hashCode() {
        return Objects.hash(d0, d1, d2, d3);
    }

}
