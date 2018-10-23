import {chunk, zip} from 'lodash';
import {IMlModel, TypeMatrix} from '../model-interfaces';
import math from '../utils/MathExtra';

import * as tfc from '@tensorflow/tfjs-core';

const { isMatrix } = math.contrib;

interface StrNumDict {
  [key: string]: ReadonlyArray<ReadonlyArray<number>>;
}

interface InterfaceFitModel<T> {
  classCategories: ReadonlyArray<T> ,
  mean: tfc.Tensor<tfc.Rank>,
  variance: tfc.Tensor<tfc.Rank>,
}

interface InterfaceFitModelAsArray<T> {
  classCategories: ReadonlyArray<T> ,
  mean: ReadonlyArray<number>,
  variance: ReadonlyArray<number>,
}

const SQRT_2PI = Math.sqrt(Math.PI * 2);

/**
 * The Naive is an intuitive method that uses probabilistic of each attribute
 * being in each class to make a prediction. It uses Gaussian function to estimate
 * probability of a given class.
 *
 * @example
 * import { GaussianNB } from 'kalimdor/naive_bayes';
 *
 * const nb = new GaussianNB();
 * const X = [[1, 20], [2, 21], [3, 22], [4, 22]];
 * const y = [1, 0, 1, 0];
 * nb.fit({ X, y });
 * nb.predict({ X: [[1, 20]] }); // returns [ 1 ]
 *
 */
export class GaussianNB<T extends number | string = number>
    implements IMlModel<T> {
  private _modelState:InterfaceFitModel<T>;

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
  public predict(
    {
      X = null
    }: {
      X: TypeMatrix<number>;
    } = {
      X: null
    }
  ): T[] {
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

  public fromJSON(modelState:InterfaceFitModelAsArray<T>): void {
    console.dir(modelState);
    this._modelState = {
      classCategories: modelState.classCategories,
      mean: tfc.tensor1d(modelState.mean as number[]),
      variance: tfc.tensor1d(modelState.variance as number[]),
    };
    this._modelState.mean.print();
    this._modelState.variance.print();
  }

  /**
   * Returns a model checkpoint
   */
  public toJSON():InterfaceFitModelAsArray<T> {
    this._modelState.mean.print();
    this._modelState.variance.print();
    return {
      classCategories: this._modelState.classCategories,
      mean: [...this._modelState.mean.clone().dataSync()],
      variance: [...this._modelState.variance.clone().dataSync()],
    };
  }

  /**
   * Make a prediction
   * @param X - new data to test
   */
  private singlePredict(X:ReadonlyArray<number>): T {
    const matrixX:tfc.Tensor<tfc.Rank> = tfc.tensor1d(X as number[], 'float32');
    const numFeatures = matrixX.shape[0];

    // Comparing input and summary shapes
    const summaryLength = this._modelState.mean.shape[1];
    if (numFeatures !== summaryLength) {
      throw new Error(`Prediction input ${matrixX.shape[0]} length must be equal or less than summary length ${summaryLength}`);
    }

    const mean = this._modelState.mean.clone();
    const variance = this._modelState.variance.clone();

    const meanValPow:tfc.Tensor<tfc.Rank> = matrixX.sub(mean)
        .pow(tfc.scalar(2)).mul(tfc.scalar(-1));

    const exponent:tfc.Tensor<tfc.Rank> = meanValPow.div(variance.mul(tfc.scalar(2))).exp()
    const innerDiv:tfc.Tensor<tfc.Rank> = tfc.scalar(SQRT_2PI).mul(variance.sqrt());
    const probabilityArray:tfc.Tensor<tfc.Rank> = tfc.scalar(1)
      .div(innerDiv)
      .mul(exponent);

    const allProbabilities = chunk(probabilityArray.dataSync(), numFeatures)
      .map(probabilitySet => probabilitySet.reduce((r, p) => r * p, 1));
    probabilityArray.dispose();

    const selectionIndex = tfc.tensor1d(allProbabilities, 'float32').argMax().dataSync()[0];

    return this._modelState.classCategories[selectionIndex];
  }


  /**
   * Summarise the dataset per class using "probability density function"
   */
  private fitModel(X:TypeMatrix<number>, y:ReadonlyArray<T>):InterfaceFitModel<T> {
    const classCategories:ReadonlyArray<T> = [...(new Set(y))];

    // Separates X by classes specified by y argument
    const separatedByCategory:StrNumDict =
        zip<ReadonlyArray<number>, T>(X, y).reduce((groups, [row, category]) => {
          groups[category.toString()] = groups[category.toString()] || [];
          groups[category.toString()].push(row);

          return groups;
        }, {});;

    const modelData = classCategories.map((category:T):ReadonlyArray<ReadonlyArray<number>> => {
      return separatedByCategory[category.toString()];
    });
    const modelDataTensor:tfc.Tensor<tfc.Rank.R3> = tfc.tensor3d(modelData as number[][][], null, 'float32');
    const moments = tfc.moments(modelDataTensor, [1]);
    // TODO check for NaN or 0 variance
    // setTimeout(() => {
    //   if ([...variance.dataSync()].some(i => i === 0)) {
    //     console.error('No variance on one of the features. Errors may result.');
    //   }
    // }, 100);

    return {
      classCategories,
      ...moments,
    };
  }
}
