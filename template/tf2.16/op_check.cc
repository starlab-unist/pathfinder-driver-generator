#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;

int main() {
  LOG(WARNING) << OpRegistry::Global()->DebugString(false);

  return 0;
}
