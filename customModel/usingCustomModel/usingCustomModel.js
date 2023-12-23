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

const minValues = await FEATURE_RESULTS.MIN_VALUES.array();
const maxValues = await FEATURE_RESULTS.MAX_VALUES.array();

// Save min and max values to local storage for later use
localStorage.setItem("minValues", JSON.stringify(minValues));
localStorage.setItem("maxValues", JSON.stringify(maxValues));

// Create and define model
const model = tf.sequential();

//One dense layer with 1 neuron,
//input of 2 feature values representing the house size and  number of bedrooms
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));
model.summary();

// Configure model for training
async function loadModelAndPredict() {
  const MODEL_URL = "./model/my-model.json"; // Replace with the path to your model.json

  // Load the pre-trained model from the server
  const model = await tf.loadLayersModel(MODEL_URL);

  // Retrieve and parse min and max values from local storage
  const minValues = JSON.parse(localStorage.getItem("minValues"));
  const maxValues = JSON.parse(localStorage.getItem("maxValues"));

  // Check if minValues and maxValues are arrays and have the correct length
  if (!Array.isArray(minValues) || !Array.isArray(maxValues)) {
    throw new Error("minValues and maxValues must be arrays.");
  }
  if (minValues.length === 0 || maxValues.length === 0) {
    throw new Error("minValues and maxValues cannot be empty.");
  }

  // Re-create tensors from the retrieved values
  const MIN_VALUES_TENSOR = tf.tensor1d(minValues);
  const MAX_VALUES_TENSOR = tf.tensor1d(maxValues);

  // Predicting the price of a house with 750 sqft and 1 bedroom
  tf.tidy(() => {
    const newInputRaw = tf.tensor2d([[750, 1]]);
    const newInputNormalized = normalize(
      newInputRaw,
      MIN_VALUES_TENSOR,
      MAX_VALUES_TENSOR
    );

    const output = model.predict(newInputNormalized.NORMALIZED_VALUES);
    console.log("Prediction:");
    output.print(); // This will print the prediction to the console
  });

  // Clean up tensors
  MIN_VALUES_TENSOR.dispose();
  MAX_VALUES_TENSOR.dispose();
  model.dispose();
}

loadModelAndPredict().catch(console.error);
