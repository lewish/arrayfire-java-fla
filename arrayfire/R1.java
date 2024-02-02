package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.numbers.U;

/**
 * A rank 1 shape (Vector).
 */
public class R1<D0 extends Num<D0>> extends Shape<D0, U, U, U> {

    public R1(D0 d0) {
        super(d0, af.u(), af.u(), af.u());
    }
}
