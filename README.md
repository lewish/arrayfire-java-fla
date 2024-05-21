This is an experimental set of Java bindings for the [ArrayFire](https://github.com/arrayfire/arrayfire) general-purpose GPU tensor library, with support for automatic reverse mode differentiation, and some limited compile type shape type checking. You can read more about the motivation and goals of this project [here](https://lewish.io/posts/shape-typed-gpu-tensor-library-arrayfire).

The library is built using Java's Foreign Linker API which is currently in preview. This library has been build against JDK 21 and therefore **can only by used with Java 21**.

Currently, the API only supports a subset of functionality of the ArrayFire API surface, but enough to build some non-trivial neural networks.

## MNIST example

For a full working example of training a 2-layer neural network on MNIST, see the [SimpleNN.java](https://github.com/lewish/arrayfire-java-fla/blob/main/examples/mnist/SimpleNN.java).

This can be run with the following command, after installing requirements:

```
bazel run examples/mnist:SimpleNN
```

## Basic example

The following simple example demonstrates usage of the API, including autograd.

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
