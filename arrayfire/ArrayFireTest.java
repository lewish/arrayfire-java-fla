package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.datatypes.F32;
import arrayfire.numbers.B;
import arrayfire.numbers.C;
import arrayfire.numbers.U;
import org.junit.Assert;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;

import static arrayfire.ArrayFire.af;

@RunWith(JUnit4.class)
public class ArrayFireTest {

    @BeforeClass
    public static void setUp() {
        af.setBackend(Backend.CPU);
    }

    @Before
    public void setUpTest() {
        af.setSeed(0);
    }

    @Test
    public void randu() {
        af.tidy(() -> {
            var arr = af.randu(DataType.F64, af.shape(4));
            var data = af.data(arr);
            Assert.assertArrayEquals(
                    new double[]{0.6009535291510355, 0.027758798477684365, 0.9805505775568435, 0.2126322292221926},
                    data.toHeap(), 0);

        });
    }

    @Test
    public void randn() {
        af.tidy(() -> {
            var arr = af.randn(DataType.F64, af.shape(4));
            var data = af.data(arr);
            System.out.println(Arrays.toString(data.toHeap()));
            Assert.assertArrayEquals(
                    new double[]{0.46430344880342067, -0.6310730997345986, -1.056124304288019, 0.1600451392361099},
                    data.toHeap(), 0);

        });
    }

    @Test
    public void range() {
        af.tidy(() -> {
            var arr = af.range(4);
            var data = af.data(arr);
            Assert.assertArrayEquals(new int[]{0, 1, 2, 3}, data.toHeap());
        });
    }

    @Test
    public void shuffle() {
        af.tidy(() -> {
            var arr = af.create(1, 2, 3, 4, 5, 6, 7, 8).reshape(2, 4);
            var shuffled = af.shuffle(arr, af.d1);
            var data = af.data(shuffled);
            Assert.assertArrayEquals(new int[]{5, 6, 1, 2, 7, 8, 3, 4}, data.toHeap());
        });
    }

    @Test
    public void transpose() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var transpose = arr.transpose();
            Assert.assertArrayEquals(new float[]{1, 3, 2, 4}, af.data(transpose).toHeap(), 1E-5f);
        });
    }

    @Test
    public void mul() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3});
            var squared = af.mul(arr, arr);
            var squaredData = af.data(squared);
            Assert.assertArrayEquals(new float[]{1, 4, 9}, squaredData.toHeap(), 1E-5f);
        });
    }

    @Test
    public void cov() {
        // Stole this example from here: https://www.cuemath.com/algebra/covariance-matrix/
        af.tidy(() -> {
            var arr = af.create(new float[]{92, 80, 60, 30, 100, 70}).reshape(2, 3);
            var cov = af.cov(arr);
            Assert.assertArrayEquals(new float[]{448, 520, 520, 700}, af.data(cov).toHeap(), 1E-5f);
        });
    }

    @Test
    public void inverse() {
        af.tidy(() -> {
            var arr = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var inverse = af.inverse(arr);
            Assert.assertArrayEquals(new float[]{-2, 1, 1.5f, -0.5f}, af.data(inverse).toHeap(), 1E-5f);
        });
    }

    @Test
    public void matmul() {
        af.tidy(() -> {
            var left = af.create(new float[]{1, 2, 3, 4}).reshape(af.a(2), af.b(2));
            var right = af.create(new float[]{1, 2, 3, 4, 5, 6}).reshape(af.a(2), af.c(3));
            Tensor<F32, B, C, U, U> result = af.matmul(left.transpose(), right);
            Assert.assertArrayEquals(new float[]{5, 11, 11, 25, 17, 39}, af.data(result).toHeap(), 1E-5f);
        });
    }

    @Test
    public void mulBroadcast() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var tile = af.create(new float[]{1, 2});
            var result = af.mul(data, tile);
            Assert.assertArrayEquals(new float[]{1, 4, 3, 8}, af.data(result).toHeap(), 1E-5f);
        });
    }

    @Test
    public void mulScalar() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4});
            var result = af.mul(data, af.constant(2));
            Assert.assertArrayEquals(new float[]{2, 4, 6, 8}, af.data(result).toHeap(), 1E-5f);
        });
    }

    //  @Test
    //  public void sum() {
    //    af.tidy(() -> {
    //      var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2)));
    //      var result = data.flatten().sum();
    //      Assert.assertArrayEquals(new float[]{10}, af.data(result),.toHeap() 1E-5f);
    //    });
    //  }

    @Test
    public void min() {
        af.tidy(() -> {
            var data = af.create(new float[]{-5, 12, 0, 1});
            var result = data.min();
            Assert.assertArrayEquals(new float[]{-5}, af.data(result).toHeap(), 1e-5f);
        });
    }

    @Test
    public void imax() {
        af.tidy(() -> {
            var data = af.create(new float[]{-5, 12, 0, 1});
            var result = af.imax(data).indices();
            Assert.assertArrayEquals(new int[]{1}, af.data(result).toHeap());
        });
    }

    @Test
    public void imaxMatrix() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 4, 3}).reshape(2, 2);
            var result = af.imax(data).indices();
            Assert.assertArrayEquals(new int[]{1, 0}, af.data(result).toHeap());
        });
    }

    @Test
    public void sumMatrix() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8}).reshape(4, 2);
            var result = af.sum(data);
            Assert.assertArrayEquals(new float[]{10, 26}, af.data(result).toHeap(), 1E-5f);
        });
    }

    @Test
    public void slice() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var rowResult = data.index(af.seq(0, 1), af.seq(1, 1));
            Assert.assertArrayEquals(new float[]{3, 4}, af.data(rowResult).toHeap(), 1E-5f);
            var columnResult = data.index(af.seq(0, 0), af.seq(0, 1));
            Assert.assertArrayEquals(new float[]{1, 3}, af.data(columnResult).toHeap(), 1E-5f);
            var reverseResult = data.index(af.seq(1, 0, -1), af.seq(1, 0, -1));
            Assert.assertArrayEquals(new float[]{4, 3, 2, 1}, af.data(reverseResult).toHeap(), 1E-5f);
        });
    }

    @Test
    public void index() {
        af.tidy(() -> {
            var indexArray = af.create(DataType.U64, new long[]{1, 0}).reshape(2);
            var data = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var resultRows = data.index(af.seq(indexArray), af.seq(0, 1));
            Assert.assertArrayEquals(new float[]{2, 1, 4, 3}, af.data(resultRows).toHeap(), 1E-5f);
            var resultCols = data.index(af.seq(0, 1), af.seq(indexArray));
            Assert.assertArrayEquals(new float[]{3, 4, 1, 2}, af.data(resultCols).toHeap(), 1E-5f);
        });
    }

    @Test
    public void batch() {
        af.tidy(() -> {
            var data = af.create(new float[]{1, 2, 3, 4, 5}).reshape(1, 5);
            var batches = af.batch(data, 2);
            Assert.assertArrayEquals(new float[]{1, 2}, af.data(batches.get(0)).toHeap(), 1E-5f);
            Assert.assertArrayEquals(new float[]{5}, af.data(batches.get(2)).toHeap(), 1E-5f);
        });
    }

    @Test
    public void convolve2() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}).reshape(3, 3, 1, 1);
            var filters = af.create(new float[]{4, 3, 2, 1, 8, 6, 4, 2}).reshape(2, 2, 1, 2);
            var convolved = af.convolve2(input, filters);
            Assert.assertArrayEquals(new float[]{37, 47, 67, 77, 37 * 2, 47 * 2, 67 * 2, 77 * 2},
                    af.data(convolved).toHeap(), 1E-5f);
        });
    }

    @Test
    public void rotate() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var rotated = af.rotate(input, (float) Math.PI / 2.0f, InterpolationType.NEAREST);
            Assert.assertArrayEquals(new float[]{3, 1, 4, 2}, af.data(rotated).toHeap(), 1E-5f);
        });
    }

    @Test
    public void scale() {
        af.tidy(() -> {
            var input = af.create(new float[]{1, 2, 3, 4}).reshape(2, 2);
            var scaled = af.scale(input, 3, 3, InterpolationType.BILINEAR);
            Assert.assertArrayEquals(new float[]{1, 5 / 3f, 2, 7 / 3f, 3f, 10 / 3f, 3, 11 / 3f, 4},
                    af.data(scaled).toHeap(), 1E-5f);
        });
    }

    //  @Test
    //  public void allBackends() {
    //    af.tidy(() -> {
    //      var originalBackend = af.backend();
    //      try {
    //        for (var backend : af.availableBackends()) {
    //          af.setBackend(backend);
    //          System.out.println(af.deviceInfo());
    //          convolve2();
    //          mulBroadcast();
    //          matmul();
    //          convolve2();
    //        }
    //      } finally {
    //        af.setBackend(originalBackend);
    //      }
    //    });
    //  }

    @Test
    public void useAcrossScopes() {
        af.tidy(() -> {
            var arr1 = af.create(new float[]{1, 2, 3}).reshape(3);
            af.tidy(() -> {
                var arr2 = af.create(new float[]{1, 2, 3}).reshape(3);
                var squared = af.mul(arr1, arr2);
                var squaredData = af.data(squared);
                Assert.assertArrayEquals(new float[]{1, 4, 9}, squaredData.toHeap(), 1E-5f);
            });
        });
    }
}
