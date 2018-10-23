import {IMlModel, TypeMatrix} from '../model-interfaces';
import math from '../utils/MathExtra';

import * as tfc from '@tensorflow/tfjs-core';

const { isMatrix } = math.contrib;

interface InterfaceFitModel<T> {
  classCategories: ReadonlyArray<T> ,
  multinomialDist: tfc.Tensor<tfc.Rank>,
}

interface InterfaceFitModelAsArray<T> {
  classCategories: ReadonlyArray<T> ,
  multinomialDist: ReadonlyArray<number>,
}

/**
 * The Naive is an intuitive method that uses probabilistic of each attribute
 * being in each class to make a prediction. It uses Gaussian function to estimate
 * probability of a given class.
 *
 * @example
 * import { GaussianMN } from 'kalimdor/naive_bayes';
 *
 * const nb = new GaussianMN();
 * const X = [[1, 20], [2, 21], [3, 22], [4, 22]];
 * const y = [1, 0, 1, 0];
 * nb.fit({ X, y });
 * nb.predict({ X: [[1, 20]] }); // returns [ 1 ]
 *
 */
export class GaussianMN<T extends number | string = number>
    implements IMlModel<T> {
  private _modelState:InterfaceFitModel<T>;

  // Setting alpha=1 is called Laplace smoothing, while alpha<1 is called Lidstone smoothing.
  constructor(private readonly alpha = 1){}

  /**
   * Naive Bayes summary according to classes
   */

  /**
   * Fit date to build Gaussian Distribution summary
   * @param {T} X - training values
   * @param {T} y - target values
   */
  public fit(
    {
      X = null,
      y = null
    }: {
      X: TypeMatrix<number>;
      y: ReadonlyArray<T>;
    } = {
      X: null,
      y: null
    }
  ): void {
    if (!isMatrix(X)) {
      throw new Error('X must be a matrix');
    }
    if (!Array.isArray(y)) {
      throw new Error('y must be a vector');
    }
    if (X.length !== y.length) {
      throw new Error('X and y must be same in length');
    }
    try {
      this._modelState = this.fitModel(X, y);
    } catch (e) {
      throw e;
    }
  }

  /**
   * Predict multiple rows
   * @param {T[]} X - values to predict in Matrix format
   * @returns {number[]}
   */
  public predict(X: TypeMatrix<number>): T[] {
    try {
      return X.map((x):T => this.singlePredict(x));
    } catch (e) {
      if (!isMatrix(X)) {
        throw new Error('X must be a matrix');
      } else {
        throw e;
      }
    }
  }

  public *predictIterator(X: IterableIterator<IterableIterator<number>>): IterableIterator<T> {
    for (const x of X) {
      yield this.singlePredict([...x]);
    }
  }

  /**
   * Returns a model checkpoint
   */
  public toJSON():InterfaceFitModelAsArray<T> {
    this._modelState.multinomialDist.print();
    return {
      classCategories: this._modelState.classCategories,
      multinomialDist: [...this._modelState.multinomialDist.clone().dataSync()],
    };
  }

  public fromJSON(modelState:InterfaceFitModelAsArray<T>): void {
    this._modelState = {
      classCategories: modelState.classCategories,
      multinomialDist: tfc.tensor1d(modelState.multinomialDist as number[]),
    };
    this._modelState.multinomialDist.print();
  }

  /**
   * Make a prediction
   * @param X - new data to test
   */
  private singlePredict(predictRow:ReadonlyArray<number>): T {
    const matrixX:tfc.Tensor<tfc.Rank> = tfc.tensor1d(predictRow as number[], 'float32');
    const numFeatures = matrixX.shape[0];
    const summaryLength = this._modelState.multinomialDist.shape[1];

    // Comparing input and summary shapes
    if (numFeatures !== summaryLength) {
      throw new Error(`Prediction input ${matrixX.shape[0]} length must be equal or less than summary length ${summaryLength}`);
    }
    const classCount = this._modelState.classCategories.length;


    // log is imporant to use different multinomial forumla
    // instead of the factorial formula
    // The multinomial naive Bayes classifier becomes a linear
    // classifier when expressed in log-space
    const priorProbability = Math.log(1 / classCount);
    

    const fitProbabilites = this._modelState.multinomialDist.clone()
      .mul(matrixX);

    // sum(1) is summing columns
    const allProbabilities = fitProbabilites
      .sum(1)
      .add(tfc.scalar(priorProbability));

    const selectionIndex = allProbabilities.argMax().dataSync()[0];
    allProbabilities.dispose();

    return this._modelState.classCategories[selectionIndex];
  }

  /**
   * Summarise the dataset per class using "probability density function"
   */
  private fitModel(X:TypeMatrix<number>, y:ReadonlyArray<T>):InterfaceFitModel<T> {
    const classCategories:ReadonlyArray<T> = [...(new Set(y))];
    const numFeatures = X[0].length;

    const frequencyCount:tfc.Tensor<tfc.Rank.R2> = tfc.tensor2d(X as number[][], null, 'float32');

    const productReducedRow = [];
    for (const rowFrequencyCount of X) {
      productReducedRow.push(rowFrequencyCount.reduce((s, c) => s + c, 0));
    }

    // log is imporant to use different multinomial forumla
    const multinomialDist = frequencyCount.add(tfc.scalar(this.alpha))
        .div(
          tfc.tensor2d(productReducedRow as number[], [frequencyCount.shape[0], 1], 'float32')
          .add(tfc.scalar((numFeatures * this.alpha)) ))
        .log();

    return {
      classCategories,
      multinomialDist,
    };
  }
}
