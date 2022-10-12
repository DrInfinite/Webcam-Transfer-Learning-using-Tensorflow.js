/* Importing the tensorflow library and the tensorflow data library. */
import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

/* Importing the controller dataset and the ui. */
import { ControllerDataset } from './controllerDataset';
import * as ui from './ui';

/* Defining the number of classes that the model will be trained on. */
const NUM_CLASSES = 4;

/* Declaring a variable called webcam. */
let webcam;

/* Creating a new instance of the ControllerDataset class. */
const controllerDataset = new ControllerDataset(NUM_CLASSES);

/* Declaring two variables. */
let truncatedMobileNet;
let model;

/**
 * It loads the MobileNet model from the web, and returns a new model that outputs the activation of
 * the last convolutional layer
 * @returns A model with the input and output layers of the MobileNet model.
 */
async function loadTruncatedMobileNet() {
    const mobilenet = await tf.loadLayersModel(
        'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

/* Adding an example to the dataset. */
ui.setExampleHandler(async label => {
    let img = await getImage();

    controllerDataset.addExample(truncatedMobileNet.predict(img), label);

    ui.drawThumb(img, label);
    img.dispose();
})

/**
 * "The model is trained using the training data, and the trained model is used to predict the class of
 * the test data."
 */
async function train() {
    /* Checking if the controllerDataset.xs variable is null and if it is, it will throw an error
    saying that you need to add some examples before training. */
    if (controllerDataset.xs == null) {
        throw new Error('Add some examples before training!');
    }

    /* Creating a new model. */
    model = tf.sequential({
        layers: [
            tf.layers.flatten(
                { inputShape: truncatedMobileNet.outputs[0].shape.slice(1) }),
            tf.layers.dense({
                units: ui.getDenseUnits(),
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            tf.layers.dense({
                units: NUM_CLASSES,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
        ]
    });

    /* Creating a new optimizer and then it is compiling the model. */
    const optimizer = tf.train.adam(ui.getLearningRate());
    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

    /* Calculating the batch size. */
    const batchSize =
        Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
    if (!(batchSize > 0)) {
        throw new Error(
            `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
    }

    /* Fitting the model to the training data. */
    model.fit(controllerDataset.xs, controllerDataset.ys, {
        batchSize,
        epochs: ui.getEpochs(),
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
            }
        }
    });
}

/* Declaring a variable called isPredicting and setting it to false. */
let isPredicting = false;

async function predict() {
    /* Starting the pacman animation. */
    ui.isPredicting();
    /* Predicting the image. */
    while (isPredicting) {
        const img = await getImage();

        const embeddings = truncatedMobileNet.predict(img);

        const predictions = model.predict(embeddings);

        const predictedClass = predictions.as1D().argMax();
        const classId = (await predictedClass.data())[0];
        img.dispose();

        ui.predictClass(classId);
        await tf.nextFrame();
    }
    /* Stopping the pacman animation. */
    ui.donePredicting();
}

/**
 * It captures an image from the webcam, converts it to a tensor, and returns it
 * @returns A tensor of shape [1, 224, 224, 3]
 */
async function getImage() {
    const img = await webcam.capture();
    const processedImg =
        tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
}

/* Listening for a click event on the train button and then it will display a message saying that the
model is training and then it will set the isPredicting variable to false and then it will call the
train function. */
document.getElementById('train').addEventListener('click', async () => {
    ui.trainStatus('Training...');
    await tf.nextFrame();
    await tf.nextFrame();
    isPredicting = false;
    train();
});
/* Listening for a click event on the predict button and then it will start the pacman animation and
then it will set the isPredicting variable to true and then it will call the predict function. */
document.getElementById('predict').addEventListener('click', () => {
    ui.startPacman();
    isPredicting = true;
    predict();
});

/**
 * It loads the webcam, loads the MobileNet model, initializes the user interface, and then captures a
 * screenshot from the webcam and predicts the image
 */
async function init() {
    /* Trying to load the webcam and if it fails, it will display a message saying that the webcam is
    not available. */
    try {
        webcam = await tfd.webcam(document.getElementById('webcam'));
    } catch (e) {
        console.log(e);
        document.getElementById('no-webcam').style.display = 'block';
    }
    /* Loading the MobileNet model from the web, and returning a new model that outputs the activation
    of the last convolutional layer. */
    truncatedMobileNet = await loadTruncatedMobileNet();

    /* Initializing the user interface. */
    ui.init();

    /* Capturing a screenshot from the webcam and then predicting the image. */
    const screenShot = await webcam.capture();
    truncatedMobileNet.predict(screenShot.expandDims(0));
    screenShot.dispose();
}

/* Initializing the webcam and the model. */
init();