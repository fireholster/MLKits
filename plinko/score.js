const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {

  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}



function runAnalysis() {
  console.log(outputs);

  const k = 10;

  const testSetSize = 200;

  _.range(1, 3).forEach(feature => {

    const data = _.map(outputs, d => [d[feature], _.last(d)]);
    const [testSet, trainingSet] = splitDataSet(minMax(data, 3), testSetSize);

    const accuracy =
      _.chain(testSet)
        .filter(tp => runKNN(trainingSet, _.initial(tp), k) === _.last(tp))
        .size()
        .divide(testSetSize)
        .value();

    console.log('For feature = ', feature, 'Accuracy = ', accuracy);
  });

}

//CREATING TRAINING AND TEST DATA SET

function splitDataSet(data, testCount) {

  const shuffledData = _.shuffle(data);

  const testSet = _.slice(shuffledData, 0, testCount);

  const trainingSet = _.slice(shuffledData, testCount);

  return [testSet, trainingSet];
}

function distance(pointA, pointB) {

  var calculatedDistance =
    _.chain(pointA)
      .zip(pointB) //Takes two arrays, looks at index, and put in its own array
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5


  return calculatedDistance;

}

function runKNN(data, point, k) {
  //point has three values
  return _.chain(data)
    .map(d => {
      return [
        distance(_.initial(d), point), _.last(d)
      ];
    })
    .sortBy(d => d[0])
    .slice(0, k)
    .countBy(d => d[1])
    .toPairs()
    .sortBy(d => d[1])
    .last()
    .first()
    .parseInt()
    .value();

}

function minMax(data, featureCount) {

  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {

    //Column Extraction
    const column = clonedData.map(d => d[i]);

    const min = _.min(column);
    const max = _.max(column);

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }

  return clonedData;

}

