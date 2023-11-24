load("@rules_kotlin//kotlin:jvm.bzl", "kt_jvm_binary", "kt_jvm_library", "kt_jvm_test")

def _get_lib_srcs(srcs, recursive):
    return srcs if srcs != None else native.glob(
        ["**/*.kt"],
        exclude = ["**/*Test.kt", "**/*Binary.kt"],
    ) if recursive else native.glob(
        ["*.kt"],
        exclude = ["*Test.kt", "*Binary.kt"],
    )

def kt_library_auto(name = None, recursive = False, srcs = None, **kwargs):
    library_name = native.package_name().split("/")[-1] if name == None else name
    resolved_srcs = _get_lib_srcs(srcs, recursive)
    if (len(resolved_srcs) == 0):
        return
    kt_jvm_library(
        name = library_name,
        srcs = resolved_srcs,
        **kwargs
    )

def kt_test_auto(recursive = False, deps = [], **kwargs):
    library_name = native.package_name().split("/")[-1]
    for src in (native.glob(["**/*Test.kt"] if recursive else ["*Test.kt"])):
        kt_jvm_test(
            name = src[:-3],
            srcs = [src],
            test_class = ".".join(native.package_name().split("/")) + "." + ".".join(src[:-5].split("/")),
            deps = [
                ":" + library_name,
                "@maven//:junit_junit",
            ] + deps,
            **kwargs
        )

def kt_binary_auto(recursive = False, runtime_deps = [], srcs = None, no_lib = False, **kwargs):
    library_name = native.package_name().split("/")[-1]
    for src in (srcs if (srcs != None) else native.glob(["**/*Binary.kt"] if recursive else ["*Binary.kt"])):
        kt_jvm_binary(
            name = src[:-3],
            srcs = [src],
            main_class = ".".join(native.package_name().split("/")) + "." + src[:-3] + "Kt",
            runtime_deps = [] if no_lib else [
                ":" + library_name,
            ] + runtime_deps,
            **kwargs
        )
