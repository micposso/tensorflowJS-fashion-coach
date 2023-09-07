const cocoSsD = require('@tensorflow-models/coco-ssd');
const tf = require('@tensorflow/tfjs-node');
// fs enables loading an image for the file system using promises
const fs = require('fs');

// load the Coco SSD model and image at the same time
Promise.all([cocoSsD.load(), fs.readFile('image.jpg')])
  .then((results) => {
    // fist result in the Coco-SSD model object
    const model = results[0];
    // second result is an image buffer
    const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3);
    // call detect() method to run inference
    return model.detect(imgTensor);
  })
  .then((predictions) => {
    console.log(JSON.stringify(predictions, null, 2));
  });

  // Tensors are N-dimensional arrays and code data structure of tensorflowJS
