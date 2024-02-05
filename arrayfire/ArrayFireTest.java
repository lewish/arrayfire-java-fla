package arrayfire;

import arrayfire.numbers.A;
import arrayfire.numbers.B;
import arrayfire.numbers.C;
import arrayfire.numbers.D;
import arrayfire.optimizers.SGD;
import org.junit.*;
import org.junit.rules.TestRule;
import org.junit.rules.TestWatcher;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Set;

import static arrayfire.ArrayFire.*;
import static org.junit.Assert.*;

@RunWith(JUnit4.class)
public class ArrayFireTest {
    @Rule
    public TestRule watcher = new TestWatcher() {
        protected void starting(Description description) {
            System.out.println("Starting test: " + description.getMethodName());
        }
    };

    @Before
    public void setUpTest() {
        af.setBackend(Backend.CPU);
        af.setRandomEngineType(RandomEngineType.AF_RANDOM_ENGINE_PHILOX_4X32_10);
        af.setSeed(0);
    }

    @After
    public void validateMemory() {
        assertEquals(0, Scope.trackedContainers());
    }

    @Test
    public void constants() {
        af.tidy(() -> {
            assertEquals((byte) 5, (byte) af.data(af.constant((byte) 5)).get(0));
            assertEquals(5, (int) af.data(af.constant(5)).get(0));
            assertEquals(5, (long) af.data(af.constant(5L)).get(0));
            assertEquals(5f, af.data(af.constant(5f)).get(0), 0);
            assertEquals(5, af.data(af.constant(5.0)).get(0), 0);
        });
    }

    @Test
    public void hostArrayEmpty() {
        af.tidy(() -> {
            var arr = af.createHost(F32, shape(n(1)));
            arr.set(0, 5f);
            Assert.assertArrayEquals(new float[]{5}, heap(arr), 1E-5f);
        });
    }

    @Test
    public void hostArrayLiterals() {
        af.tidy(() -> {
            Assert.assertArrayEquals(new int[]{1, 2}, heap(createHost(1, 2)));
            Assert.assertArrayEquals(new float[]{1, 2}, heap(createHost(1f, 2f)), 1E-5f);
            Assert.assertArrayEquals(new double[]{1, 2}, heap(createHost(1.0, 2.0)), 1E-5f);
        });
    }

    @Test
    public void castShapes() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4});
            assertEquals(af.shape(af.a(4)), arr.castshape(af::a).shape());
            assertEquals(af.shape(af.a(4), af.b(1)), arr.castshape(af::a, af::b).shape());
            assertEquals(af.shape(af.a(4), af.b(1), af.c(1)), arr.castshape(af::a, af::b, af::c).shape());
            assertEquals(af.shape(af.a(4), af.b(1), af.c(1), af.d(1)),
                arr.castshape(af::a, af::b, af::c, af::d).shape());
        });
    }

    @Test
    public void randu() {
        af.tidy(() -> {
            var arr = af.randu(af.F64, af.shape(4));
            var data = af.data(arr);
            assertArrayEquals(
                createHost(F64, 0.6009535291510355, 0.027758798477684365, 0.9805505775568435, 0.2126322292221926),
                data);

        });
    }

    @Test
    public void randn() {
        af.tidy(() -> {
            var arr = af.randn(af.F64, af.shape(4));
            var data = af.data(arr);
            assertArrayEquals(
                createHost(F64, 0.46430344880342067, -0.6310730997345986, -1.056124304288019, 0.1600451392361099),
                data);

        });
    }

    @Test
    public void range() {
        af.tidy(() -> {
            var arr = af.range(4);
            var data = af.data(arr);
            assertArrayEquals(af.createHost(U32, 0, 1, 2, 3), data);
        });
    }

    @Test
    public void rangeF32() {
        af.tidy(() -> {
            var arr = af.range(af.F32, 4);
            var data = af.data(arr);
            assertArrayEquals(new float[]{0, 1, 2, 3}, data);
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
            assertArrayEquals(new float[]{1, 2}, af.data(arr));
        });
    }

    @Test
    public void createDouble() {
        af.tidy(() -> {
            var arr = af.create(1.0, 2.0);
            assertArrayEquals(new double[]{1, 2}, af.data(arr));
        });
    }

    @Test
    public void sort() {
        af.tidy(() -> {
            var arr = af.create(new float[]{4, 2, 1, 3});
            var sorted = af.sort(arr);
            assertArrayEquals(new float[]{1, 2, 3, 4}, af.data(sorted));
        });
    }

    @Test
    public void sortIndex() {
        af.tidy(() -> {
            var arr = af.create(new float[]{4, 44, 3, 33, 2, 22, 1, 11}).reshape(2, 4);
            var sorted = af.sortIndex(arr, af.D1);
            var values = af.data(sorted.values());
            var indices = af.data(sorted.indices());
            assertArrayEquals(new float[]{1.0f, 11.0f, 2.0f, 22.0f, 3.0f, 33.0f, 4.0f, 44.0f}, values);
            assertArrayEquals(af.createHost(U32, shape(2, 4), 3, 3, 2, 2, 1, 1, 0, 0), indices);
        });
    }

    @Test
    public void permutationIndex() {
        af.tidy(() -> {
            var arr = af.create(1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 4);
            var permutation = af.permutation(arr.shape().d1());
            var shuffled = af.index(arr, af.span(), permutation);
            var data = af.data(shuffled);
            assertArrayEquals(new int[]{5, 6, 1, 2, 7, 8, 3, 4}, data);
        });
    }

    @Test
    public void transpose() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var transpose = af.transpose(arr);
            assertArrayEquals(new float[]{1, 3, 2, 4}, af.data(transpose));
        });
    }

    @Test
    public void cov() {
        // Stole this example from here: https://www.cuemath.com/algebra/covariance-matrix/
        af.tidy(() -> {
            var arr = af.create(new float[]{92, 80, 60, 30, 100, 70}).reshape(2, 3);
            var cov = af.cov(arr);
            assertArrayEquals(new float[]{448, 520, 520, 700}, af.data(cov));
        });
    }

    @Test
    public void zca() {
        af.tidy(() -> {
            var arr = af.create(new float[]{92, 80, 60, 30, 100, 70}).reshape(2, 3);
            var zca = af.zca(arr);
            assertArrayEquals(new float[]{0.11045734f, -0.06326824f, -0.06326824f, 0.0797966f}, af.data(zca));
        });
    }

    @Test
    public void deviceMemInfo() {
        af.tidy(() -> {
            af.deviceMemInfo();
        });
    }

    @Test
    public void inverse() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var inverse = af.inverse(arr);
            assertArrayEquals(new float[]{-2, 1, 1.5f, -0.5f}, af.data(inverse));
        });
    }

    @Test
    public void matmul() {
        af.tidy(() -> {
            var left = af.create(new float[]{1, 2, 3, 4}).reshape(a(2), b(2));
            var right = af.create(new float[]{1, 2, 3, 4, 5, 6}).reshape(a(2), c(3));
            var result = af.matmul(af.transpose(left), right);
            assertArrayEquals(new float[]{5, 11, 11, 25, 17, 39}, data(result));
        });
    }

    @Test
    public void matmulS32() {
        af.tidy(() -> {
            var left = af.create(new float[]{1, 2, 3, 4}).reshape(a(2), b(2));
            var right = af.create(new float[]{1, 2, 3, 4, 5, 6}).reshape(a(2), c(3));
            var result = af.matmul(af.transpose(left), right);
            assertArrayEquals(new float[]{5, 11, 11, 25, 17, 39}, data(result));
        });
    }

    @Test
    public void svd() {
        af.tidy(() -> {
            var a = af.a(2);
            var b = af.b(3);
            var matrix = af.create(F32, new float[]{1, 2, 3, 4, 5, 6}).reshape(a, b);
            var svd = af.svd(matrix);
            var u = svd.u(); // Array<F32, A, A, U, U>
            var s = svd.s(); // Array<F32, A, U, U, U>
            var vt = svd.vt(); // Array<F32, B, B, U, U>
            // Recreate the matrix from the SVD.
            var recreated = af.matmul(u, af.diag(s), af.index(vt, af.seq(a))); // Array<F32, A, B, U, U>
            assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, data(recreated), 1E-5);
        });
    }

    @Test
    public void mul() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2});
            var tile = af.create(new float[]{1, 2});
            var result = af.mul(data, tile);
            assertArrayEquals(new float[]{1, 4}, af.data(result));
        });
    }

    @Test(expected = IllegalArgumentException.class)
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
            assertArrayEquals(new float[]{1, 4, 3, 8}, af.data(result));
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
    public void exp() {
        af.tidy(() -> {
            var data = af.create(new float[]{0, 1});
            var result = af.exp(data);
            assertArrayEquals(new float[]{1, 2.7182817f}, af.data(result));
            var gradient = af.grads(result, data);
            assertArrayEquals(new float[]{1, 2.7182817f}, af.data(gradient));
        });
    }

    @Test
    public void abs() {
        af.tidy(() -> {
            var data = af.create(new float[]{-1, 2});
            var result = af.abs(data);
            assertArrayEquals(new float[]{1, 2}, af.data(result));
            var gradient = af.grads(result, data);
            assertArrayEquals(new float[]{-1, 1}, af.data(gradient));
        });
    }

    @Test
    public void mulScalar() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.mul(data, af.constant(2f).tile());
            assertArrayEquals(new float[]{2, 4, 6, 8}, af.data(result));
        });
    }

    @Test
    public void min() {
        af.tidy(() -> {
            var data = af.create(new float[]{-5, 12, 0, 1});
            var result = af.min(data);
            assertArrayEquals(new float[]{-5}, af.data(result));
        });
    }

    @Test
    public void imax() {
        af.tidy(() -> {
            var data = af.create(new float[]{-5, 12, 0, 1});
            var result = af.imax(data).indices();
            assertArrayEquals(new int[]{1}, af.data(result.cast(S32)));
        });
    }

    @Test
    public void imaxMatrix() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 4, 3}).reshape(2, 2);
            var result = af.imax(data).indices();
            assertArrayEquals(new int[]{1, 0}, af.data(result.cast(S32)));
        });
    }

    @Test
    public void topk() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 4, 3});
            var result = af.topk(data, af.k(2));
            assertArrayEquals(new float[]{4, 3}, af.data(result.values()));
            assertArrayEquals(new int[]{2, 3}, af.data(result.indices().cast(S32)));
        });
    }

    @Test
    public void sum() {
        af.tidy(() -> {
            var data = af
                           .create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
                           .reshape(2, 2, 2, 2);
            assertArrayEquals(new float[]{3, 7, 11, 15, 19, 23, 27, 31}, af.data(af.sum(data)));
            assertArrayEquals(new float[]{4, 6, 12, 14, 20, 22, 28, 30}, af.data(af.sum(data, af.D1)));
            assertArrayEquals(new float[]{6, 8, 10, 12, 22, 24, 26, 28}, af.data(af.sum(data, af.D2)));
            assertArrayEquals(new float[]{10, 12, 14, 16, 18, 20, 22, 24}, af.data(af.sum(data, af.D3)));
        });
    }

    @Test
    public void sumB8() {
        af.tidy(() -> {
            var data = af.create(U8, new byte[]{1, 2, 3, 4}).reshape(2, 2);
            var sum = af.sum(data);
            assertArrayEquals(new int[]{3, 7}, af.data(sum.cast(S32)));
        });
    }

    @Test
    public void mean() {
        af.tidy(() -> {
            var data = af
                           .create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
                           .reshape(2, 2, 2, 2);

            assertArrayEquals(new float[]{1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f, 15.5f}, af.data(af.mean(data)));
            assertArrayEquals(new float[]{2, 3, 6, 7, 10, 11, 14, 15}, af.data(af.mean(data, af.D1)));
            assertArrayEquals(new float[]{3, 4, 5, 6, 11, 12, 13, 14}, af.data(af.mean(data, af.D2)));
            assertArrayEquals(new float[]{5, 6, 7, 8, 9, 10, 11, 12}, af.data(af.mean(data, af.D3)));
        });
    }


    @Test
    public void slice() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var rowResult = af.index(data, af.seq(0, 1), af.seq(1, 1));
            assertArrayEquals(new float[]{3, 4}, af.data(rowResult));
            var columnResult = af.index(data, af.seq(0, 0), af.seq(0, 1));
            assertArrayEquals(new float[]{1, 3}, af.data(columnResult));
            var reverseResult = af.index(data, af.seq(1, 0, -1), af.seq(1, 0, -1));
            assertArrayEquals(new float[]{4, 3, 2, 1}, af.data(reverseResult));
        });
    }

    @Test
    public void index() {
        af.tidy(() -> {
            var indexArray = af.create(af.U64, new long[]{1, 0}).reshape(2);
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var resultRows = af.index(data, af.seq(indexArray), af.seq(0, 1));
            assertArrayEquals(new float[]{2, 1, 4, 3}, af.data(resultRows));
            var resultCols = af.index(data, af.seq(0, 1), af.seq(indexArray));
            assertArrayEquals(new float[]{3, 4, 1, 2}, af.data(resultCols));
        });
    }

    @Test
    public void index4D() {
        af.tidy(() -> {
            var data = af
                           .create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
                           .reshape(2, 2, 2, 2);
            Array<F32, Shape<A, B, C, D>> result = af.index(data, af.seq(af.create(0).reshape(af.a(1))),
                af.seq(af.create(1).reshape(af.b(1))), af.seq(af.create(0).reshape(af.c(1))),
                af.seq(af.create(1).reshape(af.d(1))));
            assertArrayEquals(new float[]{11}, af.data(result));
        });
    }

    @Test
    public void index3D() {
        af.tidy(() -> {
            var a = af.a(2);
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(a, a, a);
            var result = af.index(data, af.span(), af.seq(0, 0), af.seq(af.create(U32, new int[]{1})));
            assertArrayEquals(new float[]{5, 6}, af.data(result));
        });
    }

    @Test
    public void indexSpan() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(2, 2, 2);
            var result = af.index(data, af.span(), af.seq(0, 0), af.seq(0, 0));
            assertArrayEquals(new float[]{1, 2}, af.data(result));
        });
    }

    @Test
    public void zeros() {
        af.tidy(() -> {
            var data = af.zeros(F32, af.shape(2, 2));
            assertArrayEquals(new float[]{0, 0, 0, 0}, af.data(data));
        });
    }

    @Test
    public void setRandomEngineType() {
        af.tidy(() -> {
            af.setRandomEngineType(RandomEngineType.AF_RANDOM_ENGINE_MERSENNE_GP11213);
            var data = af.randu(af.F64, af.shape(1));
            assertArrayEquals(new double[]{0.4446248512515619}, af.data(data));
        });
        af.tidy(() -> {
            af.setRandomEngineType(RandomEngineType.AF_RANDOM_ENGINE_THREEFRY_2X32_16);
            var data = af.randu(af.F64, af.shape(1));
            assertArrayEquals(new double[]{0.21128287646002053}, af.data(data));
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
            var result = af.ge(data, 2f);
            assertArrayEquals(new boolean[]{false, true, true, true}, af.data(result));
        });
    }

    @Test
    public void maxof() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.maxof(data, 2f);
            assertArrayEquals(new float[]{2, 2, 3, 4}, af.data(result));
        });
    }

    @Test
    public void minof() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.minof(data, 2f);
            assertArrayEquals(new float[]{1, 2, 2, 2}, af.data(result));
        });
    }

    @Test
    public void join() {
        af.tidy(() -> {
            var data1 = af.create(1).reshape(1, 1, 1, 1);
            var data2 = af.create(2).reshape(1, 1, 1, 1);
            var join0 = af.join(data1, data2);
            assertEquals(join0.shape(), af.shape(2, 1, 1, 1));
            assertArrayEquals(new int[]{1, 2}, af.data(join0));
            var join1 = af.join(data1, data2, af.D1);
            assertEquals(join1.shape(), af.shape(1, 2, 1, 1));
            assertArrayEquals(new int[]{1, 2}, af.data(join1));
            var join2 = af.join(data1, data2, af.D2);
            assertEquals(join2.shape(), af.shape(1, 1, 2, 1));
            assertArrayEquals(new int[]{1, 2}, af.data(join2));
            var join3 = af.join(data1, data2, af.D3);
            assertEquals(join3.shape(), af.shape(1, 1, 1, 2));
            assertArrayEquals(new int[]{1, 2}, af.data(join3));
        });
    }

    @Test
    public void batch() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5}).reshape(1, 5);
            var batches = af.batch(data, 2);
            assertArrayEquals(new float[]{1, 2}, af.data(batches.get(0)));
            assertArrayEquals(new float[]{5}, af.data(batches.get(2)));
        });
    }

    @Test
    public void convolve2() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}).reshape(3, 3, 1);
            var filters = af.create(new float[]{4, 3, 2, 1, 8, 6, 4, 2}).reshape(2, 2, 1, 2);
            var convolved = af.convolve2(input, filters);
            assertArrayEquals(new float[]{37, 47, 67, 77, 37 * 2, 47 * 2, 67 * 2, 77 * 2}, af.data(convolved));
        });
    }

    @Test
    public void convolve2Padding() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2, 1);
            var filters = af.create(new float[]{4, 3, 2, 1}).reshape(2, 2, 1, 1);
            var convolved = af.convolve2(input, filters, shape(1, 1), shape(1, 1));
            assertArrayEquals(new float[]{4, 11, 6, 14, 30, 14, 6, 11, 4}, af.data(convolved));
        });
    }

    @Test
    public void convolve2Stride() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2, 1);
            var filters = af.create(new float[]{4, 3, 2, 1}).reshape(2, 2, 1, 1);
            var convolved = af.convolve2(input, filters, shape(2, 2), shape(1, 1));
            assertArrayEquals(new float[]{4, 6, 6, 4}, af.data(convolved));
        });
    }

    @Test
    public void rotate() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var rotated = af.rotate(input, (float) Math.PI / 2.0f, InterpolationType.NEAREST);
            assertArrayEquals(new float[]{3, 1, 4, 2}, af.data(rotated));
        });
    }

    @Test
    public void scale() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var scaled = af.scale(input, af.n(3), af.n(3), InterpolationType.BILINEAR);
            assertArrayEquals(new float[]{1, 5 / 3f, 2, 7 / 3f, 3f, 10 / 3f, 3, 11 / 3f, 4}, af.data(scaled), 1E-5);
        });
    }

    @Test
    public void softmax() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4});
            var softmax = af.softmax(input);
            assertArrayEquals(new float[]{0.0320586f, 0.08714432f, 0.23688284f, 0.6439143f}, af.data(softmax), 1E-5);
        });
    }

    @Test
    public void graph() {
        af.tidy(() -> {
            var left = af.create(new float[]{1, 2, 3, 4}).reshape(a(2), b(2));
            var right = af.create(new float[]{1, 2, 3, 4, 5, 6}).reshape(a(2), c(3));
            var leftT = af.transpose(left);
            var matmul = af.matmul(leftT, right);
            var log = af.log(matmul);
            var sum = af.sum(matmul);
            var graph = new Graph(af.scope().operations());
            assertEquals(Set.of(left), graph.dependencies(leftT));
            assertEquals(Set.of(leftT, right), graph.dependencies(matmul));
            assertEquals(Set.of(matmul), graph.dependencies(log));
            assertEquals(Set.of(matmul), graph.dependencies(sum));

            assertEquals(Set.of(leftT), graph.dependents(left));
            assertEquals(Set.of(log, sum), graph.dependents(matmul));
        });
    }

    @Test
    public void graphPrune() {
        af.tidy(() -> {
            var start = af.create(new float[]{1, 2, 3, 4}).reshape(a(2), b(2));
            var ignored = af.sum(start);
            var ignored2 = af.sum(ignored);
            var loss = af.sum(start);
            var graph = new Graph(af.scope().operations());
            var pruned = graph.prune(loss, start);
            assertEquals(Set.of(start, loss), pruned);
        });
    }

    @Test
    public void graphGradientsSimple() {
        af.tidy(() -> {
            var start = af.create(1.0f, 1.0f);
            var negated = af.negate(start);
            var loss = af.sum(negated);
            var startGrads = af.grads(loss, start);
            assertArrayEquals(new float[]{-1, -1}, data(startGrads));
        });
    }

    @Test
    public void graphGradientsTwoPaths() {
        af.tidy(() -> {
            var start = af.create(1.0f, 1.0f);
            var negated = af.negate(start);
            var added = af.add(start, negated);
            var startGrads = af.grads(added, start);
            assertArrayEquals(new float[]{0, 0}, af.data(startGrads));
        });
    }

    @Test
    public void gradientDescentSimpleOptimizer() {
        af.tidy(() -> {
            var a = af.params(() -> af.randu(F32, shape(n(5))), SGD.create());
            var b = af.randu(F32, shape(n(5)));
            var latestLoss = Float.POSITIVE_INFINITY;
            for (int i = 0; i < 50 && latestLoss >= 1E-10; i++) {
                latestLoss = af.tidy(() -> {
                    var mul = af.mul(a, b);
                    var loss = af.pow(af.sub(af.sum(mul), af.constant(5f)), 2);
                    af.optimize(loss);
                    return af.data(loss).get(0);
                });
            }
            assertEquals(0, latestLoss, 1E-10);
        });
    }

    @Test
    public void evalMultiple() {
        af.tidy(() -> {
            var random = af.randu(F32, shape(n(1_000_000)));
            var transform1 = af.exp(random);
            var transform2 = af.exp(transform1);
            af.eval(transform1, transform2);
        });
    }

    @Test(expected = ArrayFireException.class)
    public void useAfterRelease() {
        af.tidy(() -> {
            var arr = af.create(2f);
            var pow2 = af.pow(arr, 2);
            var pow4 = af.pow(pow2, 2);
            pow2.release();
            // Can still use pow4 after release.
            assertEquals(16, af.data(pow4).get(0), 0);
            // Throws.
            af.data(pow2);
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
                assertArrayEquals(new float[]{1, 4, 9}, squaredData);
            });
        });
    }

    @Test
    public void decodeEncodeU32() {
        var values = new int[]{0, 1, 2, Integer.MAX_VALUE};
        af.tidy(() -> {
            var result = af.data(af.create(af.S32, values));
            assertArrayEquals(values, result);
        });
    }

    public static <S extends Shape<?, ?, ?, ?>, HA extends HostArray<B8, Boolean, S>> void assertArrayEquals(
        boolean[] expected, HA actual) {
        assertArrayEquals(af.createHost(B8, actual.shape(), false, expected), actual, 0);
    }

    public static <S extends Shape<?, ?, ?, ?>, HA extends HostArray<S32, Integer, S>> void assertArrayEquals(
        int[] expected, HA actual) {
        assertArrayEquals(af.createHost(S32, actual.shape(), false, expected), actual, 0);
    }

    public static <S extends Shape<?, ?, ?, ?>, HA extends HostArray<F32, Float, S>> void assertArrayEquals(
        float[] expected, HA actual) {
        assertArrayEquals(af.createHost(F32, actual.shape(), false, expected), actual, 0);
    }

    public static <S extends Shape<?, ?, ?, ?>, HA extends HostArray<F64, Double, S>> void assertArrayEquals(
        double[] expected, HA actual) {
        assertArrayEquals(af.createHost(F64, actual.shape(), false, expected), actual, 0);
    }

    public static <S extends Shape<?, ?, ?, ?>, HA extends HostArray<F32, Float, S>> void assertArrayEquals(
        float[] expected, HA actual, double epsilon) {
        assertArrayEquals(af.createHost(F32, actual.shape(), false, expected), actual, epsilon);
    }

    public static <S extends Shape<?, ?, ?, ?>, HA extends HostArray<F64, Double, S>> void assertArrayEquals(
        double[] expected, HA actual, double epsilon) {
        assertArrayEquals(af.createHost(F64, actual.shape(), false, expected), actual, epsilon);
    }

    public static <JAT, DTM extends DataType.Meta<?, ?, JAT>, DT extends DataType<DTM>, HA extends HostArray<DT, ?, ?>> void assertArrayEquals(
        HA expected, HA actual) {
        assertArrayEquals(expected, actual, 0);
    }

    public static <JAT, DTM extends DataType.Meta<?, ?, JAT>, DT extends DataType<DTM>, HA extends HostArray<DT, ?, ?>> void assertArrayEquals(
        HA expected, HA actual, double epsilon) {
        if (!expected.shape().equals(actual.shape())) {
            throw new AssertionError("Shapes differ: " + expected.shape() + " != " + actual.shape());
        }
        if (expected.length() != actual.length()) {
            throw new AssertionError("Lengths differ: " + expected.length() + " != " + actual.length());
        }
        for (int i = 0; i < expected.length(); i++) {
            if (expected.get(i) instanceof Boolean) {
                if (expected.get(i) != actual.get(i)) {
                    throw new AssertionError("Element " + i + " differs: " + expected.get(i) + " != " + actual.get(i));
                }
            } else {
                if (Math.abs(((Number) expected.get(i)).doubleValue() - ((Number) actual.get(i)).doubleValue()) >
                        epsilon) {
                    throw new AssertionError("Element " + i + " differs: " + expected.get(i) + " != " + actual.get(i));
                }
            }
        }
    }
}
