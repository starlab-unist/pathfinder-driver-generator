load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "fuzzer_util",
    hdrs = ["fuzzer_util.h"],
    srcs = ["fuzzer_util.cc"],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:cc_ops",
    ],
)

tf_cc_binary(
    name = "op_check",
    srcs = ["op_check.cc"],
    deps = [
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:cc_ops",
    ],
)
