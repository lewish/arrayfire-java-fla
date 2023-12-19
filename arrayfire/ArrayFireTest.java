package arrayfire;

import arrayfire.datatypes.F32;
import arrayfire.numbers.A;
import arrayfire.numbers.B;
import arrayfire.numbers.C;
import arrayfire.numbers.D;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;

import static arrayfire.ArrayFire.*;
import static org.junit.Assert.*;

@RunWith(JUnit4.class)
public class ArrayFireTest {

    @Before
    public void setUpTest() {
        af.setBackend(Backend.CPU);
        af.setRandomEngineType(RandomEngineType.AF_RANDOM_ENGINE_PHILOX_4X32_10);
        af.setSeed(0);
    }

    @Test
    public void randu() {
        af.tidy(() -> {
            var arr = af.randu(af.F64, af.shape(4));
            var data = af.data(arr);
            assertArrayEquals(
                    new double[]{0.6009535291510355, 0.027758798477684365, 0.9805505775568435, 0.2126322292221926},
                    data.java(), 1E-5);

        });
    }

    @Test
    public void randn() {
        af.tidy(() -> {
            var arr = af.randn(af.F64, af.shape(4));
            var data = af.data(arr);
            System.out.println(Arrays.toString(data.java()));
            assertArrayEquals(
                    new double[]{0.46430344880342067, -0.6310730997345986, -1.056124304288019, 0.1600451392361099},
                    data.java(), 1E-5);

        });
    }

    @Test
    public void range() {
        af.tidy(() -> {
            var arr = af.range(4);
            var data = af.data(arr);
            assertArrayEquals(new int[]{0, 1, 2, 3}, data.java());
        });
    }

    @Test
    public void rangeF32() {
        af.tidy(() -> {
            var arr = af.range(af.F32, 4);
            var data = af.data(arr);
            assertArrayEquals(new float[]{0, 1, 2, 3}, data.java(), 1E-5f);
        });
    }

    @Test
    public void sync() {
        af.tidy(() -> {
            af.create(F32, 1f, 2f);
            af.sync();
        });
    }

    @Test
    public void createWithType() {
        af.tidy(() -> {
            var arr = af.create(F32, 1f, 2f);
            assertArrayEquals(new float[]{1, 2}, af.data(arr).java(), 1E-5f);
        });
    }

    @Test
    public void createDouble() {
        af.tidy(() -> {
            var arr = af.create(1.0, 2.0);
            assertArrayEquals(new double[]{1, 2}, af.data(arr).java(), 1E-5f);
        });
    }

    @Test
    public void sort() {
        af.tidy(() -> {
            var arr = af.create(new float[]{4, 2, 1, 3});
            var sorted = af.sort(arr);
            assertArrayEquals(new float[]{1, 2, 3, 4}, af.data(sorted).java(), 1E-5f);
        });
    }

    @Test
    public void sortIndex() {
        af.tidy(() -> {
            var arr = af.create(new float[]{4, 44, 3, 33, 2, 22, 1, 11}).reshape(2, 4);
            var sorted = af.sortIndex(arr, af.D1);
            var values = af.data(sorted.values());
            var indices = af.data(sorted.indices());
            assertArrayEquals(new float[]{1.0f, 11.0f, 2.0f, 22.0f, 3.0f, 33.0f, 4.0f, 44.0f}, values.java(), 1E-5f);
            assertArrayEquals(new int[]{3, 3, 2, 2, 1, 1, 0, 0}, indices.java());
        });
    }

    @Test
    public void shuffle() {
        af.tidy(() -> {
            var arr = af.create(1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 4);
            var shuffled = af.shuffle(arr, af.D1);
            var data = af.data(shuffled);
            assertArrayEquals(new int[]{5, 6, 1, 2, 7, 8, 3, 4}, data.java());
        });
    }

    @Test
    public void transpose() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var transpose = arr.transpose();
            assertArrayEquals(new float[]{1, 3, 2, 4}, af.data(transpose).java(), 1E-5f);
        });
    }

    @Test
    public void cov() {
        // Stole this example from here: https://www.cuemath.com/algebra/covariance-matrix/
        af.tidy(() -> {
            var arr = af.create(new float[]{92, 80, 60, 30, 100, 70}).reshape(2, 3);
            var cov = af.cov(arr);
            assertArrayEquals(new float[]{448, 520, 520, 700}, af.data(cov).java(), 1E-5f);
        });
    }

    @Test
    public void inverse() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var inverse = af.inverse(arr);
            assertArrayEquals(new float[]{-2, 1, 1.5f, -0.5f}, af.data(inverse).java(), 1E-5f);
        });
    }

    @Test
    public void matmul() {
        af.tidy(() -> {
            var left = af.create(new float[]{1, 2, 3, 4}).reshape(a(2), b(2));
            var right = af.create(new float[]{1, 2, 3, 4, 5, 6}).reshape(a(2), c(3));
            var result = af.matmul(left.transpose(), right);
            assertArrayEquals(new float[]{5, 11, 11, 25, 17, 39}, data(result).java(), 1E-5f);
        });
    }

    @Test
    public void matmulS32() {
        af.tidy(() -> {
            var left = af.create(new float[]{1, 2, 3, 4}).reshape(a(2), b(2));
            var right = af.create(new float[]{1, 2, 3, 4, 5, 6}).reshape(a(2), c(3));
            var result = af.matmul(left.transpose(), right);
            assertArrayEquals(new float[]{5, 11, 11, 25, 17, 39}, data(result).java(), 1E-5f);
        });
    }

    @Test
    public void svd() {
        af.tidy(() -> {
            var a = af.a(2);
            var b = af.b(3);
            var matrix = af.create(F32, new float[]{1, 2, 3, 4, 5, 6}).reshape(a, b);
            var svd = af.svd(matrix);
            var u = svd.u(); // Tensor<F32, A, A, U, U>
            var s = svd.s(); // Tensor<F32, A, U, U, U>
            var vt = svd.vt(); // Tensor<F32, B, B, U, U>
            // Recreate the matrix from the SVD.
            var recreated = af.matmul(u, af.diag(s), af.index(vt, af.seq(a))); // Tensor<F32, A, B, U, U>
            assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, data(recreated).java(), 1E-5f);
        });
    }

    @Test
    public void mul() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2});
            var tile = af.create(new float[]{1, 2});
            var result = af.mul(data, tile);
            assertArrayEquals(new float[]{1, 4}, af.data(result).java(), 1E-5f);
        });
    }

    @Test(expected = ArrayFireException.class)
    public void mulMismatched() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2});
            var tile = af.create(new float[]{1, 2, 3, 4});
            af.mul(data, tile);
        });
    }

    @Test
    public void mulTileable() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var tile = af.create(new float[]{1, 2});
            var result = af.mul(data, tile.tile());
            assertArrayEquals(new float[]{1, 4, 3, 8}, af.data(result).java(), 1E-5f);
        });
    }

    @Test(expected = IllegalArgumentException.class)
    public void mulTileableExpansion() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2});
            var tile = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            af.mul(data, tile.tile());
        });
    }


    @Test
    public void mulScalar() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.mul(data, af.constant(2).tile());
            assertArrayEquals(new float[]{2, 4, 6, 8}, af.data(result).java(), 1E-5f);
        });
    }

    @Test
    public void min() {
        af.tidy(() -> {
            var data = af.create(new float[]{-5, 12, 0, 1});
            var result = data.min();
            assertArrayEquals(new float[]{-5}, af.data(result).java(), 1e-5f);
        });
    }

    @Test
    public void imax() {
        af.tidy(() -> {
            var data = af.create(new float[]{-5, 12, 0, 1});
            var result = af.imax(data).indices();
            assertArrayEquals(new int[]{1}, af.data(result).java());
        });
    }

    @Test
    public void imaxMatrix() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 4, 3}).reshape(2, 2);
            var result = af.imax(data).indices();
            assertArrayEquals(new int[]{1, 0}, af.data(result).java());
        });
    }

    @Test
    public void sum() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}).reshape(2, 2, 2,
                    2);
            assertArrayEquals(new float[]{3, 7, 11, 15, 19, 23, 27, 31}, af.data(af.sum(data)).java(), 1E-5f);
            assertArrayEquals(new float[]{4, 6, 12, 14, 20, 22, 28, 30}, af.data(af.sum(data, af.D1)).java(), 1E-5f);
            assertArrayEquals(new float[]{6, 8, 10, 12, 22, 24, 26, 28}, af.data(af.sum(data, af.D2)).java(), 1E-5f);
            assertArrayEquals(new float[]{10, 12, 14, 16, 18, 20, 22, 24}, af.data(af.sum(data, af.D3)).java(), 1E-5f);
        });
    }

    @Test
    public void sumB8() {
        af.tidy(() -> {
            var data = af.create(U8, new byte[]{1, 2, 3, 4}).reshape(2, 2);
            var sum = af.sum(data);
            assertArrayEquals(new int[]{3, 7}, af.data(sum).java());
        });
    }

    @Test
    public void mean() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}).reshape(2, 2, 2,
                    2);

            assertArrayEquals(new float[]{1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f, 15.5f},
                    af.data(af.mean(data)).java(), 1E-5f);
            assertArrayEquals(new float[]{2, 3, 6, 7, 10, 11, 14, 15}, af.data(af.mean(data, af.D1)).java(), 1E-5f);
            assertArrayEquals(new float[]{3, 4, 5, 6, 11, 12, 13, 14}, af.data(af.mean(data, af.D2)).java(), 1E-5f);
            assertArrayEquals(new float[]{5, 6, 7, 8, 9, 10, 11, 12}, af.data(af.mean(data, af.D3)).java(), 1E-5f);
        });
    }


    @Test
    public void slice() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var rowResult = af.index(data, af.seq(0, 1), af.seq(1, 1));
            assertArrayEquals(new float[]{3, 4}, af.data(rowResult).java(), 1E-5f);
            var columnResult = af.index(data, af.seq(0, 0), af.seq(0, 1));
            assertArrayEquals(new float[]{1, 3}, af.data(columnResult).java(), 1E-5f);
            var reverseResult = af.index(data, af.seq(1, 0, -1), af.seq(1, 0, -1));
            assertArrayEquals(new float[]{4, 3, 2, 1}, af.data(reverseResult).java(), 1E-5f);
        });
    }

    @Test
    public void index() {
        af.tidy(() -> {
            var indexArray = af.create(af.U64, new long[]{1, 0}).reshape(2);
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var resultRows = af.index(data, af.seq(indexArray), af.seq(0, 1));
            assertArrayEquals(new float[]{2, 1, 4, 3}, af.data(resultRows).java(), 1E-5f);
            var resultCols = af.index(data, af.seq(0, 1), af.seq(indexArray));
            assertArrayEquals(new float[]{3, 4, 1, 2}, af.data(resultCols).java(), 1E-5f);
        });
    }

    @Test
    public void index4D() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}).reshape(2, 2, 2,
                    2);
            Tensor<F32, A, B, C, D> result = af.index(data, af.seq(af.create(0).castshape(af::a)),
                    af.seq(af.create(1).castshape(af::b)), af.seq(af.create(0).castshape(
                            af::c)), af.seq(af.create(1).castshape(af::d)));
            assertArrayEquals(new float[]{11}, af.data(result).java(), 1E-5f);
        });
    }

    @Test
    public void index3D() {
        af.tidy(() -> {
            var a = af.a(2);
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(a, a, a);
            var result = af.index(data,
                    af.span(),
                    af.seq(0, 0),
                    af.seq(af.create(U32, new int[]{1})));
            assertArrayEquals(new float[]{5, 6}, af.data(result).java(), 1E-5f);
        });
    }

    @Test
    public void indexSpan() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(2, 2, 2);
            var result = af.index(data, af.span(), af.seq(0, 0), af.seq(0, 0));
            assertArrayEquals(new float[]{1, 2}, af.data(result).java(), 1E-5f);
        });
    }

    @Test
    public void zeros() {
        af.tidy(() -> {
            var data = af.zeros(F32, af.shape(2, 2));
            assertArrayEquals(new float[]{0, 0, 0, 0}, af.data(data).java(), 1E-5f);
        });
    }

    @Test
    public void setRandomEngineType() {
        af.tidy(() -> {
            af.setRandomEngineType(RandomEngineType.AF_RANDOM_ENGINE_MERSENNE_GP11213);
            var data = af.randu(af.F64, af.shape(1));
            assertArrayEquals(new double[]{0.4446248512515619}, af.data(data).java(), 1E-5);
        });
        af.tidy(() -> {
            af.setRandomEngineType(RandomEngineType.AF_RANDOM_ENGINE_THREEFRY_2X32_16);
            var data = af.randu(af.F64, af.shape(1));
            assertArrayEquals(new double[]{0.21128287646002053}, af.data(data).java(), 1E-5);
        });
    }

    @Test
    public void backends() {
        af.tidy(() -> {
            var backends = af.availableBackends();
            assertTrue(backends.contains(Backend.CPU));
            af.setBackend(Backend.CPU);
            assertEquals(Backend.CPU, af.backend());
        });
    }

    @Test
    public void devices() {
        af.tidy(() -> {
            var originalDevice = af.deviceId();
            try {
                for (int i = 0; i < af.deviceCount(); i++) {
                    af.setDeviceId(i);
                    assertEquals(af.deviceId(), i);
                    var deviceInfo = af.deviceInfo();
                    assertFalse(deviceInfo.compute().isEmpty());
                    assertFalse(deviceInfo.platform().isEmpty());
                    assertFalse(deviceInfo.toolkit().isEmpty());
                    af.deviceGc();
                }
            } finally {
                af.setDeviceId(originalDevice);
            }
        });
    }

    @Test
    public void ge() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.ge(data, af.constant(2).tileAs(data.shape()));
            assertArrayEquals(new boolean[]{false, true, true, true}, af.data(result).java());
        });
    }

    @Test
    public void maxof() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.maxof(data, af.constant(2).tileAs(data.shape()));
            assertArrayEquals(new float[]{2, 2, 3, 4}, af.data(result).java(), 1E-5f);
        });
    }

    @Test
    public void minof() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.minof(data, af.constant(2).tileAs(data.shape()));
            assertArrayEquals(new float[]{1, 2, 2, 2}, af.data(result).java(), 1E-5f);
        });
    }

    @Test
    public void join() {
        af.tidy(() -> {
            var data1 = af.create(1).reshape(1, 1, 1, 1);
            var data2 = af.create(2).reshape(1, 1, 1, 1);
            var join0 = af.join(data1, data2);
            assertEquals(join0.shape(), af.shape(2, 1, 1, 1));
            assertArrayEquals(new int[]{1, 2}, af.data(join0).java());
            var join1 = af.join(data1, data2, af.D1);
            assertEquals(join1.shape(), af.shape(1, 2, 1, 1));
            assertArrayEquals(new int[]{1, 2}, af.data(join1).java());
            var join2 = af.join(data1, data2, af.D2);
            assertEquals(join2.shape(), af.shape(1, 1, 2, 1));
            assertArrayEquals(new int[]{1, 2}, af.data(join2).java());
            var join3 = af.join(data1, data2, af.D3);
            assertEquals(join3.shape(), af.shape(1, 1, 1, 2));
            assertArrayEquals(new int[]{1, 2}, af.data(join3).java());
        });
    }

    @Test
    public void batch() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5}).reshape(1, 5);
            var batches = af.batch(data, 2);
            assertArrayEquals(new float[]{1, 2}, af.data(batches.get(0)).java(), 1E-5f);
            assertArrayEquals(new float[]{5}, af.data(batches.get(2)).java(), 1E-5f);
        });
    }

    @Test
    public void convolve2() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}).reshape(3, 3, 1, 1);
            var filters = af.create(new float[]{4, 3, 2, 1, 8, 6, 4, 2}).reshape(2, 2, 1, 2);
            var convolved = af.convolve2(input, filters);
            assertArrayEquals(new float[]{37, 47, 67, 77, 37 * 2, 47 * 2, 67 * 2, 77 * 2}, af.data(convolved).java(),
                    1E-5f);
        });
    }

    @Test
    public void rotate() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var rotated = af.rotate(input, (float) Math.PI / 2.0f, InterpolationType.NEAREST);
            assertArrayEquals(new float[]{3, 1, 4, 2}, af.data(rotated).java(), 1E-5f);
        });
    }

    @Test
    public void scale() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var scaled = af.scale(input, af.n(3), af.n(3), InterpolationType.BILINEAR);
            assertArrayEquals(new float[]{1, 5 / 3f, 2, 7 / 3f, 3f, 10 / 3f, 3, 11 / 3f, 4}, af.data(scaled).java(),
                    1E-5f);
        });
    }

    @Test
    public void useAcrossScopes() {
        af.tidy(() -> {
            var arr1 = af.create(new float[]{1, 2, 3}).reshape(3);
            af.tidy(() -> {
                var arr2 = af.create(new float[]{1, 2, 3}).reshape(3);
                var squared = af.mul(arr1, arr2);
                var squaredData = af.data(squared);
                assertArrayEquals(new float[]{1, 4, 9}, squaredData.java(), 1E-5f);
            });
        });
    }

    @Test
    public void decodeEncodeF16() {
        var values = new float[]{0.0f, 1.0f, -1f, 100.0f, -50f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY};
        af.tidy(() -> {
            var result = af.data(af.create(af.F16, values)).java();
            Assert.assertArrayEquals(values, result, 1E-5f);
        });
    }

    @Test
    public void decodeEncodeU32() {
        var values = new int[]{0, 1, 2, Integer.MAX_VALUE};
        af.tidy(() -> {
            var result = af.data(af.create(af.U32, values)).java();
            Assert.assertArrayEquals(values, result);
        });
    }
}
