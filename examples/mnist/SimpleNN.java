package examples.mnist;

import arrayfire.af;
import arrayfire.optimizers.SGD;

/**
 * A simple 2 layer neural network for classifying MNIST digits.
 * $ bazel run examples/mnist:SimpleNN
 */
public class SimpleNN {
    public static void main(String[] args) {
        af.tidy(() -> {
            af.setSeed(0);

            var optimizer = SGD.create().learningRate(0.1f);
            var hiddenDim = af.a(2000);
            var hiddenWeights = af.params(
                () -> af.normalize(af.randn(af.F32, af.shape(af.i(Dataset.IMAGE_SIZE), hiddenDim))), optimizer);
            var weights = af.params(
                () -> af.normalize(af.randn(af.F32, af.shape(hiddenDim, af.l(Dataset.LABEL_COUNT)))), optimizer);

            Harness.run((imageBatch, labelBatch, train) -> {
                var imagesF32 = imageBatch.cast(af.F32);
                var imageNorm = af.normalize(af.center(imagesF32));
                var hidden = af.relu(af.matmul(af.transpose(hiddenWeights), imageNorm));
                var predict = af.softmax(af.matmul(af.transpose(weights), hidden));
                if (train) {
                    var labelsOneHot = af.oneHot(labelBatch.cast(af.S32), af.l(Dataset.LABEL_COUNT));
                    var rmsLoss = af.pow(af.sub(labelsOneHot, predict), 2);
                    af.optimize(rmsLoss);
                }
                return af.imax(predict).indices().cast(af.U8);
            });
        });
    }
}