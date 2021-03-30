"use strict";

async function run() {
/*     const houseSalesDataset = tf.data.csv("./kc_house_data.csv");
    console.log(houseSalesDataset); */
}

function createModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
      units: 10,
      useBias: true,
      activation: 'sigmoid',
      inputDim: 2,
    }));
    model.add(tf.layers.dense({
      units: 10,
      activation: 'sigmoid',
      useBias: true,
    }));
    // Output layer:
    model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid',
      useBias: true,
    }));
    
    return model;
}

function trainModel(model, inputs, labels) {
    //Optimizer: ADAM
    //Loss: binaryCrossentropy
    //Metrics: 'accuracy'
    //const optimizer = tf.train.adam();

}