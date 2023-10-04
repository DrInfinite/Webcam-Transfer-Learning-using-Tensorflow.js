/* Importing the TensorFlow.js library. */
import * as tf from '@tensorflow/tfjs';

/* It takes in a number of classes and adds examples to the dataset */
export class ControllerDataset {
    /**
     * The constructor function takes in a number of classes and assigns it to the numClasses variable.
     * @param numClasses - The number of classes that the model will be trained to recognize.
     */
    constructor(numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * Adds an example to the controller dataset.
     * @param {Tensor} example A tensor representing the example. It can be an image,
     *     an activation, or any other type of Tensor.
     * @param {number} label The label of the example. Should be a number.
     */
    addExample(example, label) {
        /* Converting the label into a one-hot vector. */
        const y = tf.tidy(
            () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));

        /* Checking if the xs and ys variables are null. If they are, it assigns the example and y
        variables to them. */
        if (this.xs == null) {
            this.xs = tf.keep(example);
            this.ys = tf.keep(y);
        }

        /* Adding the new example to the existing dataset. */
        else {
            const oldX = this.xs;
            this.xs = tf.keep(oldX.concat(example, 0));

            const oldY = this.ys;
            this.ys = tf.keep(oldY.concat(y, 0));

            oldX.dispose();
            oldY.dispose();
            y.dispose();
        }
    }
}