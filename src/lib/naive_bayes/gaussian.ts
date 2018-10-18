import { cloneDeep, isNaN } from 'lodash';
import { exp, mean, pi, pow, sqrt, std } from 'mathjs';
import math from '../utils/MathExtra';

const { isMatrix } = math.contrib;

type TypeMatrix<T> = ReadonlyArray<ReadonlyArray<T>>
interface StrNumDict<T> {
  [key: string]: T;
  [key: number]: T;
}
type StrNumDictArray = StrNumDict<Array<ReadonlyArray<number>>>;


interface InterfaceSummarizeByClass<T> {
  [key: string]: {
    class:T;
    dist: ReadonlyArray<[number, number]>;
  }
}

/**
 * The Naive is an intuitive method that uses probabilistic of each attribute
 * belonged to each class to make a prediction. It uses Gaussian function to estimate
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
export class GaussianNB<T extends string | number = number> {
  /**
   * Naive Bayes summary according to classes
   */
  private summaries: InterfaceSummarizeByClass<T> = null;

  /**
   * @param clone - To clone the input values during fit and predict
   */
  constructor(private clone:boolean = true) {}

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
    let clonedX = X;
    let clonedY = y;
    if (this.clone) {
      clonedX = cloneDeep(X);
      clonedY = cloneDeep(y);
    }
    this.summaries = this.summarizeByClass(clonedX, clonedY);
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
    if (!isMatrix(X)) {
      throw new Error('X must be a matrix');
    }
    // let clonedX = X;

    // if (this.clone) {
    //   clonedX = cloneDeep(X);
    // }
    // const result:T[] = [];
    // for (let i = 0; i < clonedX.length; i++) {
    //   result.push(this.singlePredict(clonedX[i]));
    // }
    return X.map(x => this.singlePredict(x));
  }

  /**
   * Restores GaussianNB model from a checkpoint
   * @param summaries - Gaussian Distribution summaries
   */
  public fromJSON(
    {
      summaries = null
    }: {
      summaries: {};
    } = {
      summaries: null
    }
  ): void {
    this.summaries = summaries;
  }

  /**
   * Returns a model checkpoint
   */
  public toJSON(): {
    summaries: {};
  } {
    return {
      summaries: this.summaries
    };
  }

  /**
   * Make a prediction
   * @param X -
   */
  private singlePredict(X:ReadonlyArray<number>): T {
    const summaryKeys:ReadonlyArray<string> = Object.keys(this.summaries);

    // Comparing input and summary shapes
    const summaryLength = this.summaries[summaryKeys[0]].dist.length;
    const inputLength = X.length;
    if (inputLength > summaryLength) {
      throw new Error('Prediction input X length must be equal or less than summary length');
    }

    // Getting probability of each class
    // TODO Log Probabilities
    const probabilities:StrNumDict<number> = {};
    for (const key of summaryKeys) {
      probabilities[key] = 1;
      const classSummary:ReadonlyArray<[number, number]> = this.summaries[key].dist;
      for (let j = 0; j < classSummary.length; j++) {
        const [meanval, stdev] = classSummary[j];
        const probability:number = this.calculateProbability(X[j], meanval, stdev);
        if (!isNaN(probability)) {
          probabilities[key] *= probability;
        }
      }
    }


    // // Vote the best predction
    const [keyOfBestClass, probOfBestClass] = Object.entries(probabilities)
        .reduce((maxEntry, [key, prob]) => maxEntry && maxEntry[1] > prob ? maxEntry:[key, prob]);

    // Calculate Class Probabilities
    // const totalProbs = Object.values(probabilities)
    //     .reduce((sum, prob) => sum + prob, 0);
    // const classProbability = probOfBestClass / totalProbs;

    return this.summaries[keyOfBestClass].class;
  }

  /**
   * Calculate the main division
   * @param x
   * @param meanval
   * @param stdev
   */
  private calculateProbability(x:number, meanval:number, stdev:number): number {
    const stdevPow:any = pow(stdev, 2);
    const meanValPow = -pow(x - meanval, 2);
    const exponent = exp(meanValPow / (2 * stdevPow));
    return (1 / (sqrt(pi.valueOf() * 2) * stdev)) * exponent;
  }

  /**
   * Summarise the dataset per class using "probability density function"
   * example:
   * Given
   * const X = [[1,20], [2,21], [3,22], [4,22]];
   * const y = [1, 0, 1, 0];
   * Returns
   * { '0': [ [ 3, 1.4142135623730951 ], [ 21.5, 0.7071067811865476 ] ],
   * '1': [ [ 2, 1.4142135623730951 ], [ 21, 1.4142135623730951 ] ] }
   * @param dataset
   */
  private summarizeByClass(X:TypeMatrix<number>, y:ReadonlyArray<T>): InterfaceSummarizeByClass<T> {
    const separated:StrNumDictArray = this.separateByClass(X, y);
    const summarize: InterfaceSummarizeByClass<T> = {};
    for (const key of Object.keys(separated)) {
      // Finding the real target value from y array
      const targetClass:T = y.find(z => z.toString() === key);
      // Mutating "separated" variable instead of immutable approach for performance
      separated[key].forEach(x => x.push(targetClass));
      const dataset = separated[key];
      // storing object to each attribute to store real class value and dist summary
      summarize[key] = {
        class: targetClass,
        dist: this.summarize(dataset)
      };
    }
    return summarize;
  }

  /**
   * Summarise the dataset to calculate the ‘pdf’ (probability density function) later on
   * @param dataset
   */
  private summarize(dataset:Array<Array<string|number>>): ReadonlyArray<[number, number]> {
    const sorted = [];
    // Manual ZIP; simulating Python's zip(*data)
    // TODO: Find a way to use a built in function
    for (let zRow = 0; zRow < dataset.length; zRow++) {
      const row = dataset[zRow];
      for (let zCol = 0; zCol < row.length; zCol++) {
        // Pushes a new array placeholder if it's not populated yet at zRow index
        if (typeof sorted[zCol] === 'undefined') {
          sorted.push([]);
        }
        const element = dataset[zRow][zCol];
        sorted[zCol].push(element);
      }
    }

    const summaries = [];
    for (let i = 0; i < sorted.length; i++) {
      const attributes: any = sorted[i];
      summaries.push([mean(attributes), std(attributes)]);
    }
    // Removing the last element
    summaries.pop();
    return summaries;
  }

  /**
   * Separates X by classes specified by y argument
   * Given
   * const X = [[1,20], [2,21], [3,22], [4,22]];
   * const y = [1, 0, 1, 0];
   * Returns
   * { '0': [ [2,21], [4,22] ],
   * '1': [ [1,20], [3,22] ] }
   * @param X
   * @param y
   */
  private separateByClass(X:TypeMatrix<number>, y:ReadonlyArray<T>):StrNumDictArray {
    const result:StrNumDictArray = {};
    for (let i = 0; i < X.length; i++) {
      const row:ReadonlyArray<number> = X[i];
      const target:string|number = y[i];
      if (result[target]) {
        // If value already exist
        result[target].push(row);
      } else {
        result[target] = [];
        result[target].push(row);
      }
    }
    return result;
  }
}