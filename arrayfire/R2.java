package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.U;

/**
 * A rank 2 shape (Matrix).
 */
public class R2<D0 extends Num<D0>, D1 extends Num<D1>> extends Shape<D0, D1, U, U> {

    public R2(D0 d0, D1 d1) {
        super(d0, d1, af.u(), af.u());
    }
}
