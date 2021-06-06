//require("@tensorflow/tfjs-node");
require("@tensorflow/tfjs-node-gpu");
const tf = require('@tensorflow/tfjs-node-gpu');
const LineRegression = require('./linear-regression');
const loadCSV = require('./load-csv');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {

    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
});

const regression = new LineRegression(features, labels, {

    learningRate: 0.1,
    iterations: 100
});

regression.features.print();

regression.train();

//with Slow
// console.log('(Without TF) Value of M is ', regression.m);
// console.log('(without TF) Value of B is ', regression.b);

//with Fast
 //console.log('(With TF) Value of M is ', regression.weights.get(1, 0));
 //console.log('(With TF) Value of B is ', regression.weights.get(0, 0));

 const r2 = regression.test(testFeatures, testLabels);

 plot({

     x : regression.bHistory,
     y: regression.historyMSE.reverse(),
     xLabel: "Value of b",
     yLabel: "MSE"
 });

 console.log('Accuracy: ', r2 + ''.red);
 console.log('Running Backend:' , tf.getBackend(), ''.red);