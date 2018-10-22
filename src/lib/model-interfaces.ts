// Nothing here yet

export type TypeMatrix<T> = ReadonlyArray<ReadonlyArray<T>> | number[][];
export type TypeVector<T> = ReadonlyArray<T> | number[];
export interface TypeInput<T> {
  X: TypeMatrix<number> | TypeVector<number>;
  y: TypeVector<T>;
}


export abstract class IMlModel<T> {
  /**
   * Fit date to build Gaussian Distribution summary
   * @param {T} X - training values
   * @param {T} y - target values
   */
  public abstract fit(data:TypeInput<T>, log?:() => IMlModel<T> | null): void;

  /**
   * Predict multiple rows
   * @param {T[]} X - values to predict in Matrix format
   * @returns {number[]}
   */
  // public abstract predict(X: TypeMatrix<number>): T[];

  /**
   * Predict multiple rows
   * @param {T[]} X - values to predict in Matrix format
   * @returns {number[]}
   */
  // public abstract predictIterator(data: {
  //     X: IterableIterator<IterableIterator<number>>;
  //   }): T[];

  /**
   * Restores GaussianNB model from a checkpoint
   * @param summaries - Gaussian Distribution summaries
   */
  public abstract fromJSON(json: any): void;
  /**
   * Returns a model checkpoint
   */
  public abstract toJSON(): any;
}

export class Serialization<T> {
  protected _modelState:T;

  /**
   * Restores GaussianNB model from a checkpoint
   * @param modelState - Gaussian Distribution modelState
   */
  public fromJSON(modelState): void {
    this._modelState = modelState;
  }

  /**
   * Returns a model checkpoint
   */
  public toJSON():any {
    return this._modelState;
  }
}