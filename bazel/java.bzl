def java_library_auto(recursive = False, srcs = None, **kwargs):
    library_name = native.package_name().split("/")[-1]
    native.java_library(
        name = library_name,
        srcs = srcs if srcs != None else native.glob(
            ["**/*.java"],
            exclude = ["**/*Test.java", "**/*Binary.java"],
        ) if recursive else native.glob(
            ["*.java"],
            exclude = ["*Test.java", "*Binary.java"],
        ),
        **kwargs
    )

def java_test_auto(recursive = False, deps = [], **kwargs):
    library_name = native.package_name().split("/")[-1]
    for src in (native.glob(["**/*Test.java"] if recursive else ["*Test.java"])):
        native.java_test(
            name = src[:-5],
            srcs = [src],
            test_class = ".".join(native.package_name().split("/")) + "." + ".".join(src[:-5].split("/")),
            deps = [
                ":" + library_name,
                "@maven//:junit_junit",
            ] + deps,
            **kwargs
        )

def java_binary_auto(recursive = False, deps = [], srcs = None, **kwargs):
    library_name = native.package_name().split("/")[-1]
    for src in (srcs if (srcs != None) else native.glob(["**/*Binary.java"] if recursive else ["*Binary.java"])):
        native.java_binary(
            name = src[:-5],
            srcs = [src],
            main_class = ".".join(native.package_name().split("/")) + "." + src[:-5],
            deps = [
                ":" + library_name,
            ] + deps,
            **kwargs
        )
