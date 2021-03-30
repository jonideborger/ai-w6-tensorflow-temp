"use strict";

//Follow along
//https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#0

//Dataset
//https://storage.googleapis.com/tfjs-tutorials/carsData.json
//Feature: Horesepower
//Label: Miles per gallon

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
  
/*     tfvis.render.scatterplot(
      {name: 'Horsepower v MPG'},
      {values}, 
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    ); */
  
    //const model = createModel();  
    //tfvis.show.modelSummary({name: 'Model Summary'}, model);

    //const tensorData = convertToTensor(data);
    //const {inputs, labels} = tensorData;
        
    //await trainModel(model, inputs, labels);
    //console.log('Done Training');
}
  
async function getData() {
    //const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');  
    //const carsData = await carsDataResponse.json();  
}

function createModel() {
    // Create a sequential model
    const model = tf.sequential(); 
    
    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
    
    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));
  
    return model;
}

//Pay Attention to:
//1.Shuffle
//2.Normalization
//3.Size
function convertToTensor(data) {
    
    return tf.tidy(() => {
      // Step 1. Shuffle the data    
      tf.util.shuffle(data);
  
      // Step 2. Convert data to Tensor
      const inputs = data.map(d => d.horsepower);
      const labels = data.map(d => d.mpg);
  
      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);
  
      //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();  
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
  
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
  
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });  
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.  
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    const batchSize = 32;
    const epochs = 50;
    
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
}

run();