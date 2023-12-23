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

// Create and define model
const model = tf.sequential();

//One dense layer with 1 neuron,
//input of 2 feature values representing the house size and  number of bedrooms
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));
model.summary();

// Configure model for training
async function train() {
  console.log("Training...");
  const LEARNING_RATE = 0.01; // Choosing a learning rate

  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  // Finally train the model
  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      validationSplit: 0.2, // 20% of the data will be used for validation
      shuffle: true, // Shuffle data
      batchSize: 64, // Batch size lots of data hence 64
      epochs: 10, // Go over data 10 times
    }
  );

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();
  console.log(
    "Average Loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    "Average Validation Error Loss: " +
      Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );

  evaluate();
}

async function evaluate() {
  // Predicting the price of 1 hard coded value ( a house with 750 sqft and 1 bedroom)
  tf.tidy(function () {
    let newInput = normalize(
      tf.tensor2d([[750, 1]]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );

    let output = model.predict(newInput.NORMALIZED_VALUES);

    // Prints the predicted price of the house
    output.print();
  });

  // await model.save("downloads://my-model");
  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();
  console.log(tf.memory().numTensors);
}

train();
