package arrayfire;

public record ArrayPair<L extends Array<?, ?>, R extends Array<?, ?>>(L left, R right) {

}
