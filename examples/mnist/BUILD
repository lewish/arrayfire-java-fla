java_library(
    name = "mnist",
    srcs = glob(["*.java"]),
    data = [
        "@mnist_test_images//file",
        "@mnist_test_labels//file",
        "@mnist_train_images//file",
        "@mnist_train_labels//file",
    ],
    deps = [
        "//arrayfire",
    ],
)

java_binary(
    name = "SimpleNN",
    main_class = "examples.mnist.SimpleNN",
    runtime_deps = [":mnist"],
)

java_binary(
    name = "ConvNN",
    main_class = "examples.mnist.ConvNN",
    runtime_deps = [":mnist"],
)
