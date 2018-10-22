import { concat, countBy, find, head, isEqual, keys, map, maxBy, range, reduce, values } from 'lodash';
import { IMlModel} from '../model-interfaces';
import { DecisionTreeClassifier } from '../tree/tree';

/**
 * Base RandomForest implementation used by both classifier and regressor
 * @ignore
 */
export class BaseRandomForest {
  protected trees = [];
  protected nEstimator;
  protected randomState = null;

  /**
   *
   * @param {number} nEstimator - Number of trees.
   * @param random_state - Random seed value for DecisionTrees
   */
  constructor(
    {
      // Each object param default value
      nEstimator = 10,
      random_state = null
    }: {
      // Param types
      nEstimator?: number;
      random_state?: number;
    } = {
      // Default value on empty constructor
      nEstimator: 10,
      random_state: null
    }
  ) {
    this.nEstimator = nEstimator;
    this.randomState = random_state;
  }

  /**
   * Build a forest of trees from the training set (X, y).
   * @param {Array} X - array-like or sparse matrix of shape = [n_samples, n_features]
   * @param {Array} y - array-like, shape = [n_samples] or [n_samples, n_outputs]
   * @returns void
   */
  public fit({ X = [], y = [] }: { X: number[][]; y: number[] }): void {
    this.trees = reduce(
      range(0, this.nEstimator),
      sum => {
        const tree = new DecisionTreeClassifier({
          featureLabels: null,
          randomise: true,
          random_state: this.randomState
        });
        tree.fit({ X, y });
        return concat(sum, [tree]);
      },
      []
    );
  }

  /**
   * Returning the current model's checkpoint
   * @returns {{trees: any[]; nEstimator: number}}
   */
  public toJSON(): { trees: any[]; nEstimator: number } {
    return {
      nEstimator: this.nEstimator,
      trees: this.trees
    };
  }

  /**
   * Restore the model from a checkpoint
   * @param {any[]} trees - Decision trees
   */
  public fromJSON({ trees = null }: { trees: any[] }): void {
    if (!trees) {
      throw new Error('You must provide both tree to restore the model');
    }
    this.trees = trees;
  }

  /**
   * Internal predict function used by either RandomForestClassifier or Regressor
   * @param X
   * @private
   */
  protected _predict(X: number[] | number[][] = []): any[] {
    return map(this.trees, tree => {
      // TODO: Check if it's a matrix or an array
      return tree.predict({ X });
    });
  }
}

/**
 * Random forest classifier creates a set of decision trees from randomly selected subset of training set.
 * It then aggregates the votes from different decision trees to decide the final class of the test object.
 *
 * @example
 * import { RandomForestClassifier } from 'kalimdor/ensemble';
 *
 * const X = [[0, 0], [1, 1], [2, 1], [1, 5], [3, 2]];
 * const y = [0, 1, 2, 3, 7];
 *
 * const randomForest = new RandomForestClassifier();
 * randomForest.fit({ X, y });
 *
 * // Results in a value such as [ '0', '2' ].
 * // Predictions will change as we have not set a seed value.
 */
export class RandomForestClassifier extends BaseRandomForest implements IMlModel<any> {
  /**
   * Predict class for X.
   *
   * The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates.
   * That is, the predicted class is the one with highest mean probability estimate across the trees.
   * @param {Array} X - array-like or sparse matrix of shape = [n_samples]
   * @returns {string[]}
   */
  public predict(X: number[] | number[][] = []): any[] {
    const predictions = this._predict(X);
    return this.votePredictions(predictions);
  }

  /**
   * @hidden
   * Bagging prediction helper method
   * According to the predictions returns by the trees, it will select the
   * class with the maximum number (votes)
   * @param {Array<any>} predictions - List of initial predictions that may look like [ [1, 2], [1, 1] ... ]
   * @returns {string[]}
   */
  private votePredictions(predictions: any[]): string[] {
    const counts = countBy(predictions, x => x);
    const countsArray = reduce(
      keys(counts),
      (sum, k) => {
        const returning = {};
        returning[k] = counts[k];
        return concat(sum, returning);
      },
      []
    );
    const max = maxBy(countsArray, x => head(values(x)));
    const key = head(keys(max));
    // Find the actual class values from the predictions
    return find(predictions, pred => {
      return isEqual(pred.join(','), key);
    });
  }
}
