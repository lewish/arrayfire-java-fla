package examples.mnist;

import arrayfire.HostArray;
import arrayfire.Shape;
import arrayfire.U8;
import arrayfire.af;
import arrayfire.numbers.I;
import arrayfire.numbers.N;
import arrayfire.numbers.U;

import java.io.FileInputStream;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;

public record Dataset(HostArray<U8, Byte, Shape<I, N, U, U>> images, HostArray<U8, Byte, Shape<U, N, U, U>> labels) {

    public static int TOTAL_COUNT = 70000;
    public static int LABEL_COUNT = 10;
    public static int IMAGE_SIZE = 28 * 28;
    public static int IMAGE_WIDTH = 28;
    public static int IMAGE_HEIGHT = 28;

    public static Dataset load() {
        var runFiles = System.getenv().get("JAVA_RUNFILES");
        var images = getImages(
            List.of(readGzipBytes(Path.of(runFiles, "mnist_train_images/file/downloaded").toString()),
                readGzipBytes(Path.of(runFiles, "mnist_test_images/file/downloaded").toString())));
        var labels = getLabels(
            List.of(readGzipBytes(Path.of(runFiles, "mnist_train_labels/file/downloaded").toString()),
                readGzipBytes(Path.of(runFiles, "mnist_test_labels/file/downloaded").toString())));
        return new Dataset(images, labels);
    }

    private static byte[] readGzipBytes(String path) {
        try {
            var fis = new FileInputStream(path);
            var gis = new GZIPInputStream(fis);
            return gis.readAllBytes();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static HostArray<U8, Byte, Shape<I, N, U, U>> getImages(List<byte[]> datas) {
        var shape = af.shape(af.i(IMAGE_SIZE), af.n(TOTAL_COUNT));
        var imageIndex = 0;
        var images = af.createHost(af.U8, shape);
        for (byte[] data : datas) {
            int byteIndex = 8;
            int rows = ((data[byteIndex + 3] & 0xFF)) | ((data[byteIndex + 2] & 0xFF) << 8) |
                           ((data[byteIndex + 1] & 0xFF) << 16) | ((data[byteIndex] & 0xFF) << 24);
            byteIndex += 4;
            int cols = ((data[byteIndex + 3] & 0xFF)) | ((data[byteIndex + 2] & 0xFF) << 8) |
                           ((data[byteIndex + 1] & 0xFF) << 16) | ((data[byteIndex] & 0xFF) << 24);
            byteIndex += 4;
            if (rows != 28 || cols != 28) {
                throw new IllegalStateException(String.format("Expected 28x28 but rows/cols where %s/%s", rows, cols));
            }
            while (byteIndex < data.length) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        images.set(shape.offset(i * rows + j, imageIndex), data[byteIndex]);
                        byteIndex++;
                    }
                }
                imageIndex++;
            }
        }
        return images;
    }

    private static HostArray<U8, Byte, Shape<U, N, U, U>> getLabels(List<byte[]> datas) {
        var shape = af.shape(af.U, af.n(TOTAL_COUNT));
        var labels = af.createHost(af.U8, shape);
        var labelIndex = 0;
        for (byte[] data : datas) {
            for (int i = 8; i < data.length; i++) {
                labels.set(labelIndex, data[i]);
                labelIndex++;
            }
        }
        for (int i = 0; i < labels.length(); i++) {
            if (!(labels.get(i) <= 9 && labels.get(i) >= 0)) {
                throw new IllegalStateException(
                    String.format("Label greater than 9 or less than 0: %s", labels.get(i)));
            }
        }
        return labels;
    }
}