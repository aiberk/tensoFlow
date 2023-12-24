// import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";
const LEARNING_RATE = 0.0001;
const TEST_INPUT = 10;
const BATCHSIZE = 2;
const EPOCH = 120;
const title2 = document.getElementById("title2");
let mult = TEST_INPUT * TEST_INPUT;
title2.innerText = "Solution: " + mult;
const INPUTS = [];
for (let n = 0; n <= 20; n++) {
  INPUTS.push(n);
}

const OUTPUTS = [];
for (let n = 0; n < INPUTS.length; n++) {
  OUTPUTS.push(INPUTS[n] * INPUTS[n]);
}

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor1d(INPUTS);
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

// INPUTS_TENSOR.dispose();
// OUTPUTS_TENSOR.dispose();

// Create and define model
const model = tf.sequential();

//One dense layer with 1 neuron,
//input of 1 feature values of integers squared into 3 neurons
model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: "relu" }));
//Hidden layer
model.add(tf.layers.dense({ units: 100, activation: "relu" }));
// Output neuron
model.add(tf.layers.dense({ units: 1 }));

model.summary();

// Configure model for training
async function train() {
  model.compile({
    optimizer: OPTIMIZER,
    loss: "meanSquaredError",
  });

  // Finally train the model
  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      callbacks: { onEpochEnd: logProgress },
      shuffle: true, // Shuffle data
      batchSize: BATCHSIZE, // Batch size lots of data hence 64
      epochs: EPOCH, // Go over data 10 times
    }
  );

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();
  console.log(
    "Average Loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );

  evaluate();
}

async function evaluate() {
  // Predicting the price of 1 hard coded value ( a house with 750 sqft and 1 bedroom)
  tf.tidy(function () {
    let newInput = normalize(
      tf.tensor1d([TEST_INPUT]),
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
function logProgress(epoch, logs) {
  // console.log("Epoch: " + epoch + " Loss: " + Math.sqrt(logs.loss));
  if (epoch % 70 == 0) {
    OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
  }
}

// Choosing a learning rate
const OPTIMIZER = tf.train.sgd(LEARNING_RATE); // Stochastic gradient descent
train();
