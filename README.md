This is an experimental set of Java bindings for the [ArrayFire](https://github.com/arrayfire/arrayfire) general-purpose GPU tensor library.

The library is built using Java's Foreign Linker API which is currently in preview. This library has been build against JDK 21 and therefore **can only by used with Java 21**.

Currently, the API only supports a subset of functionality of ArrayFire, along with some additional features that are geared towards machine learning applications, including automatic reverse mode differentiation.

## Basic example

The following example demonstrates usage of the API, including autograd.

```java
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
```

## Requirements

- ArrayFire version >= 3.8.0 installed
- Java version 21 exactly
- The following JVM flags to be set: `--enable-native-access=ALL-UNNAMED --enable-preview`

## Developing

This repository can be build with Bazel/[Bazelisk](https://github.com/bazelbuild/bazelisk?tab=readme-ov-file#installation). After installing Bazel/Bazelisk, run the following command to run all tests:

```
bazel test ...
```

To build the jar directly:

```
bazel build arrayfire
```
