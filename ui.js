/* Importing the tensorflow library. */
import * as tf from '@tensorflow/tfjs';

/* Defining the controls for the game. */
const CONTROLS = ['up', 'down', 'left', 'right'];
const CONTROL_CODES = [38, 40, 37, 39];

/**
 * It's a function that takes a string as an argument and returns a function that takes a string as an
 * argument and so on
 */
export function init() {
    document.getElementById('controller').style.display = '';
    statusElement.style.display = 'none';
}

/* Getting the element with the id 'train-status' from the HTML file. */
const trainStatusElement = document.getElementById('train-status');

/* Getting the element with the id 'learningRate' from the HTML file. */
const learningRateElement = document.getElementById('learningRate');
export const getLearningRate = () => +learningRateElement.value;

/* Getting the element with the id 'batchSizeFraction' from the HTML file. */
const batchSizeFractionElement = document.getElementById('batchSizeFraction');
export const getBatchSizeFraction = () => +batchSizeFractionElement.value;

/* Getting the element with the id 'epochs' from the HTML file. */
const epochsElement = document.getElementById('epochs');
export const getEpochs = () => +epochsElement.value;

/* Getting the element with the id 'dense-units' from the HTML file. */
const denseUnitsElement = document.getElementById('dense-units');
export const getDenseUnits = () => +denseUnitsElement.value;
const statusElement = document.getElementById('status');

/**
 * When the page loads, start the game.
 */
export function startPacman() {
    google.pacman.startGameplay();
}

/**
 * It takes the classId as an argument and then uses the classId to get the corresponding keyCode from
 * the CONTROL_CODES object. Then it calls the keyPressed function from the pacman.js file and passes
 * the keyCode as an argument. Finally, it sets the data-active attribute of the body to the
 * corresponding CONTROLS value
 * @param classId - The class ID of the prediction.
 */
export function predictClass(classId) {
    google.pacman.keyPressed(CONTROL_CODES[classId]);
    document.body.setAttribute('data-active', CONTROLS[classId]);
}

/**
 * The function isPredicting() is used to make the status element visible.
 *
 * The function donePredicting() is used to make the status element hidden.
 *
 * The function trainStatus() is used to set the inner text of the train status element.
 */
export function isPredicting() {
    statusElement.style.visibility = 'visible';
}
export function donePredicting() {
    statusElement.style.visibility = 'hidden';
}
export function trainStatus(status) {
    trainStatusElement.innerText = status;
}

/* Setting the addExampleHandler to the handler function. */
export let addExampleHandler;
export function setExampleHandler(handler) {
    addExampleHandler = handler;
}

/* Setting the mouseDown variable to false and the totals variable to an array of four zeros. */
let mouseDown = false;
const totals = [0, 0, 0, 0];

/* Getting the elements with the ids 'up', 'down', 'left' and 'right' from the HTML file. */
const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');

/* An empty object. */
const thumbDisplayed = {};

/**
 * It takes a label as an argument, sets the mouseDown variable to true, sets the className variable to
 * the value of the label in the CONTROLS object, sets the button variable to the element with the id
 * of the className variable, sets the total variable to the element with the id of the className
 * variable plus '-total', and then runs a while loop that runs as long as the mouseDown variable is
 * true.
 *
 * Inside the while loop, it runs the addExampleHandler function with the label as an argument, sets
 * the data-active attribute of the body element to the value of the label in the CONTROLS object,
 * increments the value of the total variable by 1, and then waits for the next frame.
 *
 * After the while loop, it removes the data-active attribute from the body element
 * @param label - The label of the example.
 */
async function handler(label) {
    mouseDown = true;
    const className = CONTROLS[label];
    const button = document.getElementById(className);
    const total = document.getElementById(className + '-total');
    while (mouseDown) {
        addExampleHandler(label);
        document.body.setAttribute('data-active', CONTROLS[label]);
        total.innerText = ++totals[label];
        await tf.nextFrame();
    }
    document.body.removeAttribute('data-active');
}

/* Adding an event listener to the up button. When the mouse is pressed down, the handler function is
called with the argument 0. When the mouse is released, the mouseDown variable is set to false. */
upButton.addEventListener('mousedown', () => handler(0));
upButton.addEventListener('mouseup', () => mouseDown = false);

/* Adding an event listener to the down button. When the mouse is pressed down, the handler function is
called with the argument 1. When the mouse is released, the mouseDown variable is set to false. */
downButton.addEventListener('mousedown', () => handler(1));
downButton.addEventListener('mouseup', () => mouseDown = false);

/* Adding an event listener to the left button. When the mouse is pressed down, the handler function is
called with the argument 2. When the mouse is released, the mouseDown variable is set to false. */
leftButton.addEventListener('mousedown', () => handler(2));
leftButton.addEventListener('mouseup', () => mouseDown = false);

/* Adding an event listener to the right button. When the mouse is pressed down, the handler function
is called with the argument 3. When the mouse is released, the mouseDown variable is set to false. */
rightButton.addEventListener('mousedown', () => handler(3));
rightButton.addEventListener('mouseup', () => mouseDown = false);

/**
 * If the image has not been displayed, draw it on the canvas.
 * @param img - the image to draw
 * @param label - The label of the control.
 */
export function drawThumb(img, label) {
    if (thumbDisplayed[label] == null) {
        const thumbCanvas = document.getElementById(CONTROLS[label] + '-thumb');
        draw(img, thumbCanvas);
    }
}

/**
 * It takes a tensor of shape [224, 224, 3] and draws it on a canvas
 * @param image - The image tensor to draw.
 * @param canvas - The canvas element to draw the image on.
 */
export function draw(image, canvas) {
    const [width, height] = [224, 224];
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    /* Converting the image data to a format that can be displayed on the canvas. */
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
        imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
        imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}