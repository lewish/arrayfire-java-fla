package examples.mnist;

import arrayfire.Array;
import arrayfire.Shape;
import arrayfire.U8;
import arrayfire.af;
import arrayfire.numbers.I;
import arrayfire.numbers.N;
import arrayfire.numbers.U;
import arrayfire.optimizers.SGD;
import arrayfire.utils.Functions;

import java.util.stream.IntStream;

/**
 * A simple 2 layer neural network for classifying MNIST digits.
 *   $ bazel run examples/mnist:SimpleNN
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

            run((imageBatch, labelBatch, train) -> {
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


    public static void run(
        Functions.Function3<Array<U8, Shape<I, N, U, U>>, Array<U8, Shape<U, N, U, U>>, Boolean, Array<U8, Shape<U, N, U, U>>> fn) {
        var mnist = Dataset.load();
        // Sort images and labels.
        var permutation = af.permutation(af.n(Dataset.TOTAL_COUNT));
        var images = af.index(af.create(mnist.images()), af.span(), permutation);
        var labels = af.index(af.create(mnist.labels()), af.span(), permutation);
        // Split into train and test sets.
        var trainImages = af.index(images, af.span(), af.seq(0, 60000 - 1));
        var trainLabels = af.index(labels, af.span(), af.seq(0, 60000 - 1));
        var testImages = af.index(images, af.span(), af.seq(60000, 70000 - 1));
        var testLabels = af.index(labels, af.span(), af.seq(60000, 70000 - 1));

        var epochs = 50;
        var batchSize = 256;

        IntStream.range(0, epochs).forEach(epoch -> {
            var trainImageBatches = af.batch(trainImages, af.D1, batchSize);
            var trainLabelBatches = af.batch(trainLabels, af.D1, batchSize);
            // Train.
            var trainCorrect = IntStream.range(0, trainImageBatches.size()).mapToLong(i -> af.tidy(() -> {
                var trainImagesBatch = trainImageBatches.get(i).get();
                var trainLabelsBatch = trainLabelBatches.get(i).get();
                var predicted = fn.apply(trainImagesBatch, trainLabelsBatch, true);
                var correct = af.sum(af.eq(predicted, trainLabelsBatch).flatten());
                return af.data(correct).get(0);
            })).sum();
            // Test.
            var testCorrect = af.tidy(() -> {
                var testImageBatches = af.batch(testImages, af.D1, batchSize);
                var testLabelBatches = af.batch(testLabels, af.D1, batchSize);
                return IntStream.range(0, testImageBatches.size()).mapToLong(i -> af.tidy(() -> {
                    var testImagesBatch = testImageBatches.get(i).get();
                    var testLabelsBatch = testLabelBatches.get(i).get();
                    var predicted = fn.apply(testImagesBatch, af.zeros(testLabelsBatch.type(), testLabelsBatch.shape()),
                        false);
                    var correct = af.sum(af.eq(predicted, testLabelsBatch).flatten());
                    return af.data(correct).get(0);
                })).sum();
            });
            System.out.printf("Epoch %s: Train: %.5f, Test: %.5f%n", epoch,
                trainCorrect / (double) trainImages.shape().d1().size(),
                testCorrect / (double) testImages.shape().d1().size());
        });

    }
}