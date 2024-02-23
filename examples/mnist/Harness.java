package examples.mnist;

import arrayfire.Array;
import arrayfire.Shape;
import arrayfire.U8;
import arrayfire.af;
import arrayfire.numbers.I;
import arrayfire.numbers.N;
import arrayfire.numbers.U;
import arrayfire.utils.Functions;

import java.util.stream.IntStream;

public class Harness {

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
