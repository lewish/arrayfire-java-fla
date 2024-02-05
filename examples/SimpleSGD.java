import arrayfire.optimizers.SGD;

import static arrayfire.af.*;

void main() {
    tidy(() -> {
        var a = params(() -> randu(F32, shape(5)), SGD.create());
        var b = randu(F32, shape(5));
        var latestLoss = Float.POSITIVE_INFINITY;
        for (int i = 0; i < 50 || latestLoss >= 1E-10; i++) {
            latestLoss = tidy(() -> {
                var mul = mul(a, b);
                var loss = pow(sub(sum(mul), 5), 2);
                optimize(loss);
                return data(loss).get(0);
            });
        }
        System.out.println(latestLoss);
    });
}