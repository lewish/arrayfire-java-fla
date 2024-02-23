package examples.mnist;

import arrayfire.Backend;
import arrayfire.af;
import arrayfire.optimizers.SGD;

/**
 * A simple 2 layer neural network for classifying MNIST digits.
 * $ bazel run examples/mnist:SimpleNN
 */
public class ConvNN {
    public static void main(String[] args) {
        af.tidy(() -> {
            af.setSeed(0);
            var optimizer = SGD.create().learningRate(0.01f);
            af.setBackend(Backend.CUDA);
            var filtersWidth = af.x(7);
            var filtersHeight = af.y(7);
            var filtersSize = af.z(filtersWidth.size() * filtersHeight.size());
            var filtersCount = af.f(64);

            var filterWeights = af.params(() -> af
                .normalize(af.randn(af.F32, af.shape(filtersSize, filtersCount)))
                .reshape(filtersWidth, filtersHeight, filtersCount), optimizer);

            var convOutputShape = af.convolve2Shape(af.shape(Dataset.IMAGE_WIDTH, Dataset.IMAGE_HEIGHT),
                af.shape(filtersWidth, filtersHeight), af.shape(1, 1), af.shape(0, 0), af.shape(1, 1));
            var poolOutputShape = af.convolve2Shape(convOutputShape, af.shape(2, 2), af.shape(1, 1), af.shape(0, 0),
                af.shape(1, 1));
            var finalWeights = af.params(() -> af.normalize(af.randn(af.F32,
                af.shape(poolOutputShape.capacity() * filtersCount.size(), af.l(Dataset.LABEL_COUNT)))), optimizer);

            Harness.run((imageBatch, labelBatch, train) -> {
                var imagesF32 = imageBatch.cast(af.F32);
                var imageNorm = af.normalize(imagesF32);
                var image2D = imageNorm.reshape(af.w(Dataset.IMAGE_WIDTH), af.h(Dataset.IMAGE_HEIGHT), af.u(),
                    imageNorm.shape().d1());
                var conv = af.relu(
                    af.convolve2(image2D, filterWeights.reshape(filtersWidth, filtersHeight, af.u(), filtersCount)));
                var pool = af.meanPool(conv, af.shape(2, 2), af.shape(1, 1), af.shape(0, 0));
                var predict = af.softmax(af.matmul(af.transpose(finalWeights),
                    pool.reshape(poolOutputShape.capacity() * pool.shape().d2().size(), pool.shape().d3().size())));
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