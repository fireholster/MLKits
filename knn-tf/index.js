require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');


function knn(features, labels, predictionPoint, k) {

    //We will take the mean and variance for the whole feature set and 
    //then use it to standardize each item.
    const { mean, variance } = tf.moments(features, 0);

    //sqrt of a variance is standard deviation
    //Standardization
    // (Value - Average or Mean) / (Std Deviation or sqrt of variance)

    const standardizedPredictionPoint = predictionPoint.sub(mean).div(variance.pow(0.5));

    return features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(standardizedPredictionPoint) //get the difference
        .pow(2) //square the value
        .sum(1) //without param all values will be summed to a single value
        .pow(0.5) //sqrt of the result. this will give the distance b/w prediction point and each feature
        //Not we have to sort the result in conjunction with labels [distance, labels]
        //our labels have shape [4, 1] while the results above is [4]
        //in order to concat. we need both labels and result to be of same shape
        .expandDims(1) //expanding dimensions of the result
        .concat(labels, 1)   //joining the results (distance) with labels on y axis (at row level)        
        .unstack() //This will take each tensor from above result, put them in sepratate tensors and whole thing in a javascript array
        .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k;
}

let { features, labels, testFeatures, testLabels } =
    loadCSV('kc_house_data.csv', {
        shuffle: true,
        splitTest: 10,
        dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
        labelColumns: ['price']
    });


features = tf.tensor(features);
labels = tf.tensor(labels);

let logResults = []
testFeatures.forEach((testFeature, i) => {

    const result = knn(features, labels, tf.tensor(testFeature), 10);
    const err = (testLabels[i][0] - result) / testLabels[i][0];
    logResults.push(
        { 'Guessed Value': result, 'Actual': testLabels[i][0], 'Accuracy (%)': err * 100 }
    );

});

console.table(logResults);

