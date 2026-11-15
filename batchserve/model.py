"""Deterministic synthetic model for measuring the serving system."""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

FEATURES = 128
HIDDEN = 768
OUTPUTS = 32


def weights() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(2417)
    dimensions = ((FEATURES, HIDDEN), (HIDDEN, HIDDEN), (HIDDEN, OUTPUTS))
    return tuple(
        rng.standard_normal((inputs, outputs), dtype=np.float32) / np.float32(inputs**0.5)
        for inputs, outputs in dimensions
    )


def model_bytes() -> bytes:
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "W1"], ["hidden1"]),
            helper.make_node("Relu", ["hidden1"], ["relu1"]),
            helper.make_node("MatMul", ["relu1", "W2"], ["hidden2"]),
            helper.make_node("Relu", ["hidden2"], ["relu2"]),
            helper.make_node("MatMul", ["relu2", "W3"], ["embedding"]),
        ],
        "synthetic-mlp",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["batch", FEATURES])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", OUTPUTS])],
        [numpy_helper.from_array(weight, name=f"W{i}") for i, weight in enumerate(weights(), 1)],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model.SerializeToString()


class OnnxModel:
    def __init__(self, threads: int) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            model_bytes(), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self(np.zeros((1, FEATURES), dtype=np.float32))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        result, = self.session.run(["embedding"], {"features": features})
        return result
