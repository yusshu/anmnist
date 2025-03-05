/* eslint-disable */
import Matrix from "@/lib/Matrix";
import {ActivationFunction} from "@/lib/activation_functions";

class Layer {
  public previousLayer: Layer | null = null;
  public nextLayer: Layer | null = null;

  public weights: Matrix | null = null;
  public biases: Matrix | null = null;

  constructor(
    public readonly size: number,
    public readonly activationFunction: ActivationFunction,
  ) {
  }
}

interface NeuralNetworkOptions {
  learningRate: number;
}

export default class NeuralNetwork {
  private layers: Layer[] = [];

  constructor(
    public readonly inputSize: number,
    public readonly options: NeuralNetworkOptions = {
      learningRate: 0.1
    }
  ) {
  }

  addLayer(size: number, activationFunction: ActivationFunction) {
    const layer = new Layer(size, activationFunction);
    if (this.layers.length > 0) {
      const previousLayer = this.layers[this.layers.length - 1];
      layer.previousLayer = previousLayer;
      previousLayer.nextLayer = layer;
    }
    this.layers.push(layer);
  }

  randomizeParams() {
    for (let layer of this.layers) {
      layer.weights = new Matrix(layer.size, layer.previousLayer?.size || this.inputSize);
      layer.biases = new Matrix(1, layer.size);
      layer.weights.fillRandom(-0.5, 0.5);
      layer.biases.fillRandom(-0.5, 0.5);
    }
  }

  run(input: Matrix) {
    let current = input;
    for (let layer of this.layers) {
      current = layer.activationFunction(layer.weights!.mul(current).add(layer.biases!.transpose()));
    }
    return current;
  }

  train(input: Matrix, output: Matrix) {
    // forward propagation
    const zs: Matrix[] = [];
    const activations: Matrix[] = [ input ];

    for (let i = 0; i < this.layers.length; i++) {
      const layer = this.layers[i];

      // z = mx + b
      const z = layer.weights!.mul(activations[i]).add(layer.biases!.transpose());
      zs.push(z);

      // a = sigma(z)
      const a = layer.activationFunction(z);
      activations.push(a);
    }

    // backpropagation
    let dCda = activations[activations.length - 1]
      .sub(output)
      .mul(1 / (output.m * output.n));

    for (let i = this.layers.length - 1; i >= 0; i--) {
      const layer = this.layers[i];

      const dCdb = dCda.hadamard(layer.activationFunction.derivative(zs[i]));
      const dCdw = dCdb.mul(activations[i].transpose());

      // backpropagate
      dCda = layer.weights!.transpose().mul(dCdb);

      // adjust parameters
      layer.weights = layer.weights!.sub(dCdw.mul(this.options.learningRate));
      layer.biases = layer.biases!.sub(dCdb.mul(this.options.learningRate).transpose());
    }
  }
}
