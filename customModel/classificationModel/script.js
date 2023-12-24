import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";
const LEARNING_RATE = 0.0001;
const TEST_INPUT = 10;
const BATCHSIZE = 512;
const EPOCH = 50;
const INPUTS = TRAINING_DATA.inputs;

const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// Create and define model
const model = tf.sequential();

//One dense layer with 1 neuron,
//input of 1 feature values of integers squared into 3 neurons
model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
//Hidden layer
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

let CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");
function drawImage(digit) {
  var imageData = CTX.getImageData(0, 0, 28, 28);
  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255; // Red channel
    imageData.data[i * 4 + 1] = digit[i] * 255; // Red channel
    imageData.data[i * 4 + 2] = digit[i] * 255; // Red channel
    imageData.data[i * 4 + 3] = 255; // Red channel
  }

  CTX.putImageData(imageData, 0, 0);
  setTimeout(evaluate, interval);
}

// // Configure model for training
async function train() {
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Finally train the model
  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true, // Shuffle data
    validationSplit: 0.2, // 20% of data used for validation
    batchSize: BATCHSIZE, // Batch size lots of data hence 64
    epochs: EPOCH, // Go over data 10 times
    callbacks: { onEpochEnd: logProgress },
  });

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  console.log(
    "Average Loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );

  evaluate();
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

async function evaluate() {
  //Select a random input from the input data
  const OFFSET = Math.floor(Math.random() * INPUTS.length);

  let answer = tf.tidy(function () {
    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();

    let output = model.predict(newInput);

    // Prints the predicted price of the house
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then(function (index) {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index == OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

function logProgress(epoch, logs) {
  // console.log("Epoch: " + epoch + " Loss: " + Math.sqrt(logs.loss));
}
train();
// // Choosing a learning rate
// const OPTIMIZER = tf.train.sgd(LEARNING_RATE); // Stochastic gradient descent
// train();
