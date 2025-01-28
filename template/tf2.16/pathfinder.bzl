load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

def pathfinder_fuzz_driver(name):
    cc_library(
        name = name,
        srcs = [
            name + ".cc",
        ],
        deps = [
            "//tensorflow/core/kernels/pathfinder:fuzzer_util",
            "//tensorflow/cc:scope",
            "//tensorflow/core:core_cpu",
            "//tensorflow/core:tensorflow",
            "//tensorflow/cc:cc_ops",
            "@pathfinder_deps//:pathfinder",
            "@pathfinder_deps//:z3",
        ],
    )
