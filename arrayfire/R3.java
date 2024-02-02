package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.U;

/**
 * A rank 3 shape.
 */
public class R3<D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>> extends Shape<D0, D1, D2, U> {

    public R3(D0 d0, D1 d1, D2 d2) {
        super(d0, d1, d2, af.u());
    }
}
