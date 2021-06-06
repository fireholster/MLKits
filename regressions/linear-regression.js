const tf = require('@tensorflow/tfjs-node-gpu');
const colors = require('colors');
const _ = require('lodash');

class LinearRegression {

    //Slow GD Constructor
    /*
    constructor(features, labels, options) {
        
        this.features = features;
        this.labels = labels;
        this.options = options;
 
        //If we want to ensure we surely get some properties back in this.options. 
        //i.e this will defaults the mandatory properties, if they are not provided.
        this.options = Object.assign(
            {
                learningRate: 0.1,
                iterations: 1000
            }, options);
 
        this.m = 0;
        this.b = 0;
    }
    */

    //Fast GD Constructor
    constructor(features, labels, options) {

        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.options = options;


        //If we want to ensure we surely get some properties back in this.options. 
        //i.e this will defaults the mandatory properties, if they are not provided.
        this.options = Object.assign(
            {
                learningRate: 0.01,
                iterations: 100
            }, options);

        // this.m = 0;
        // this.b = 0;

        //Creating and intitializing weights tensor with intial values of zero 
        //similar to what we did with this.m= 0 & this.b = 0
        this.weights = tf.zeros([this.features.shape[1], 1]);

        //used for learning rate optimization
        this.historyMSE = [];
        this.bHistory = [];
    }

    slowGradientDescent() {

        //create a set of guesses for m *x + b
        let currentGuessesForMPG = this.features.map(d => { return this.m * d[0] + this.b });

        //slope with respect to B
        //dMSE/db = 2/n (SUM( (m*x + b) - Actual)
        const bSlope =
            (_.sum(
                currentGuessesForMPG.map((d, i) => {
                    return (d - this.labels[i][0]);
                })) * 2) / this.features.length;


        //slope with respect to M
        //dMSE/dm = 2/n (SUM( ( -x * (Actual - (mx + b) )
        const mSlope =
            (_.sum(
                currentGuessesForMPG.map((d, i) => {
                    return -1 * this.features[i][0] * (this.labels[i][0] - d)
                })) * 2) / this.features.length;

        //Updating the value of our guesses to get optiomal value
        //using equation as newm = m - slope * learningRate
        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;
    }

    fastGradientDescent() {

        //M * X 
        const currentGuesses = this.features.matMul(this.weights);
        //M * X - labels
        const differences = currentGuesses.sub(this.labels);

        const slopes =
            this.features
                .transpose()
                .matMul(differences)
                .div(this.features.shape[0]);

        //Updating the value of our guesses to get optimal value
        //using equation as newm = m - slope * learningRate
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));


    }

    train() {

        for (let i = 0; i < this.options.iterations; i++) {
            
            this.bHistory.push(this.weights.arraySync()[0]);
            //this.slowGradientDescent();
            this.fastGradientDescent();

            //Saving MSE for learning rate optimization during training phase
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    test(testFeatures, testLabels) {

        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        //We learned weights as part of train. 
        //i.e. we know the values of m & x
        const predictions = testFeatures.matMul(this.weights);

        //Testing Accuracy.

        //SS(res)
        const ssResidual =
            testLabels.sub(predictions)
                .pow(2)
                .sum()
                .arraySync();
        //.get();

        const ssTotal =
            testLabels.sub(testLabels.mean())
                .pow(2)
                .sum()
                .arraySync();
        //.get();


        //Coefficient of determianation (R2)
        return 1 - (ssResidual / ssTotal);
    }

    processFeatures(features) {

        features = tf.tensor(features);

        //Doing standardization
        if (this.mean && this.variance)
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        else {
            features = this.standrdize(features);
        }

        //Doing this after standardization to avoid applying standardization on the column of ones 

        //Adding a new tensor with value of 1
        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    standrdize(features) {

        const { mean, variance } = tf.moments(features, 0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    recordMSE() {
        const mse =
            this.features.matMul(this.weights)
                .sub(this.labels)
                .pow(2)
                .sum()
                .div(this.features.shape[1])
                .arraySync();

        //this.historyMSE.push(mse); 
        //OR
        this.historyMSE.unshift(mse);
    }

    updateLearningRate() {

        //Ensure we have enough runs done for mse history to update learning rate.
        if (this.historyMSE.length < 2) {
            return;
        }

        //const lastValue = this.historyMSE[this.historyMSE.length - 1];
        //const secondLastValue = this.historyMSE[this.historyMSE.length - 2];
        //OR
        //Using unshifted store

        if (this.historyMSE[0] > this.historyMSE[1]) {
            this.options.learningRate /= 2;
        }
        else {
            this.options.learningRate *= 1.05;
        }
    }
}


module.exports = LinearRegression;