package arrayfire;

import arrayfire.numbers.U;

/**
 * A rank 0 shape (Scalar).
 */
public class R0 extends Shape<U, U, U, U> {

    public R0() {
        super(af.u(), af.u(), af.u(), af.u());
    }
}
