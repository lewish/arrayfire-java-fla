package arrayfire;

import arrayfire.datatypes.F32;
import arrayfire.numbers.B;
import arrayfire.numbers.C;
import arrayfire.numbers.U;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import static arrayfire.ArrayFire.af;

@RunWith(JUnit4.class)
public class ArrayFireTest {

    @BeforeClass
    public static void setUp() {
        af.setBackend(AfBackend.CPU);
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
            var data = af.push(af.hostTensor(new float[]{1, 2, 3, 4}, af.shape(4)));
            var result = af.mul(data, af.constant(2));
            Assert.assertArrayEquals(new float[]{2, 4, 6, 8}, af.data(result).toHeap(), 1E-5f);
        });
    }

    //  @Test
    //  public void sum() {
    //    af.tidy(() -> {
    //      var data = af.push(af.hostTensor(new float[]{1, 2, 3, 4}, af.shape(2, 2)));
    //      var result = data.flatten().sum();
    //      Assert.assertArrayEquals(new float[]{10}, af.data(result),.toHeap() 1E-5f);
    //    });
    //  }

    @Test
    public void min() {
        af.tidy(() -> {
            var data = af.push(af.hostTensor(new float[]{-5, 12, 0, 1}, af.shape(4)));
            var result = data.min();
            Assert.assertArrayEquals(new float[]{-5}, af.data(result).toHeap(), 1e-5f);
        });
    }

    @Test
    public void imax() {
        af.tidy(() -> {
            var data = af.push(af.hostTensor(new float[]{-5, 12, 0, 1}, af.shape(4)));
            var result = data.imax();
            Assert.assertArrayEquals(new int[]{1}, af.datau32(result));
        });
    }

    @Test
    public void imaxMatrix() {
        af.tidy(() -> {
            var data = af.push(af.hostTensor(new float[]{1, 2, 4, 3}, af.shape(2, 2)));
            var result = data.imax();
            Assert.assertArrayEquals(new int[]{1, 0}, af.datau32(result));
        });
    }

    @Test
    public void sumMatrix() {
        af.tidy(() -> {
            var data = af.push(af.hostTensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, af.shape(4, 2)));
            var result = data.sum();
            Assert.assertArrayEquals(new float[]{10, 26}, af.data(result).toHeap(), 1E-5f);
        });
    }

    @Test
    public void slice() {
        af.tidy(() -> {
            var data = af.push(af.hostTensor(new float[]{1, 2, 3, 4}, af.shape(2, 2)));
            var rowResult = data.index(af.seq(0, 1), af.seq(1, 1));
            Assert.assertArrayEquals(new float[]{3, 4}, af.dataf32(rowResult), 1E-5f);
            var columnResult = data.index(af.seq(0, 0), af.seq(0, 1));
            Assert.assertArrayEquals(new float[]{1, 3}, af.dataf32(columnResult), 1E-5f);
            var reverseResult = data.index(af.seq(1, 0, -1), af.seq(1, 0, -1));
            Assert.assertArrayEquals(new float[]{4, 3, 2, 1}, af.dataf32(reverseResult), 1E-5f);
        });
    }

    @Test
    public void index() {
        af.tidy(() -> {
            var indexArray = af.push(af.hostTensor(new long[]{1, 0}, af.shape(2)));
            var data = af.push(af.hostTensor(new float[]{1, 2, 3, 4}, af.shape(2, 2)));
            var resultRows = data.index(af.seq(indexArray), af.seq(0, 1));
            Assert.assertArrayEquals(new float[]{2, 1, 4, 3}, af.dataf32(resultRows), 1E-5f);
            var resultCols = data.index(af.seq(0, 1), af.seq(indexArray));
            Assert.assertArrayEquals(new float[]{3, 4, 1, 2}, af.dataf32(resultCols), 1E-5f);
        });
    }

    @Test
    public void batch() {
        af.tidy(() -> {
            var data = af.push(af.hostTensor(new float[]{1, 2, 3, 4, 5}, af.shape(1, 5)));
            var batches = af.batch(data, 2);
            Assert.assertArrayEquals(new float[]{1, 2}, af.dataf32(batches.get(0)), 1E-5f);
            Assert.assertArrayEquals(new float[]{5}, af.dataf32(batches.get(2)), 1E-5f);
        });
    }

    @Test
    public void convolve2() {
        af.tidy(() -> {
            var input = af.push(af.hostTensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, af.shape(3, 3, 1, 1)));
            var filters = af.push(af.hostTensor(new float[]{4, 3, 2, 1, 8, 6, 4, 2}, af.shape(2, 2, 1, 2)));
            var convolved = af.convolve2(input, filters);
            Assert.assertArrayEquals(new float[]{37, 47, 67, 77, 37 * 2, 47 * 2, 67 * 2, 77 * 2}, af.dataf32(convolved),
                    1E-5f);
        });
    }

    @Test
    public void rotate() {
        af.tidy(() -> {
            var input = af.hostTensor(new float[]{1, 2, 3, 4}, af.shape(2, 2)).push();
            var rotated = af.rotate(input, (float) Math.PI / 2.0f, InterpolationType.NEAREST);
            Assert.assertArrayEquals(new float[]{3, 1, 4, 2}, af.dataf32(rotated), 1E-5f);
        });
    }

    @Test
    public void scale() {
        af.tidy(() -> {
            var input = af.hostTensor(new float[]{1, 2, 3, 4}, af.shape(2, 2)).push();
            var scaled = af.scale(input, 3, 3, InterpolationType.BILINEAR);
            Assert.assertArrayEquals(new float[]{1, 5 / 3f, 2, 7 / 3f, 3f, 10 / 3f, 3, 11 / 3f, 4}, af.dataf32(scaled),
                    1E-5f);
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
            var arr1 = af.push(af.hostTensor(new float[]{1, 2, 3}, af.shape(3)));
            af.tidy(() -> {
                var arr2 = af.push(af.hostTensor(new float[]{1, 2, 3}, af.shape(3)));
                var squared = af.mul(arr1, arr2);
                var squaredData = af.dataf32(squared);
                Assert.assertArrayEquals(new float[]{1, 4, 9}, squaredData, 1E-5f);
            });
        });
    }
}
