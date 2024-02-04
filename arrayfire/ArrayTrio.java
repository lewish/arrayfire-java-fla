package arrayfire;

public record ArrayTrio<T1 extends Array<?, ?>, T2 extends Array<?, ?>, T3 extends Array<?, ?>>(T1 left, T2 middle,
                                                                                                T3 right) {

}
