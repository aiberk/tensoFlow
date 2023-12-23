import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";
const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor, min, max) {
  const result = tf.tidy(function () {
    const MIN_VALUES = min || tensor.min(0);
    const MAX_VALUES = max || tensor.max(0);

    const TENSOR_SUBTRACT_MIN_VALUE = tensor.sub(MIN_VALUES);

    const RANGE_SIZE = MAX_VALUES.sub(MIN_VALUES);
    const NORMALIZED_VALUES = TENSOR_SUBTRACT_MIN_VALUE.div(RANGE_SIZE);
    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });
  return result;
}

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log("Normalized Values");
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min Values");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max Values");
FEATURE_RESULTS.MAX_VALUES.print();

INPUTS_TENSOR.dispose();
OUTPUTS_TENSOR.dispose();
