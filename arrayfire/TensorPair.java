package arrayfire;

public record TensorPair<L extends Tensor<?, ?>, R extends Tensor<?, ?>>(L left, R right) {

}
