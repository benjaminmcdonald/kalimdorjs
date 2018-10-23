import { GaussianNB } from '../../src/lib/naive_bayes';
import { IMlModel } from '../../src/types/model-interfaces';

/*
import numpy as np
X = np.array([[1, 20], [20, 210], [3, 22], [40, 220], [6, 10], [7, 11]])
y = np.array([1, 0, 1, 0, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
print(clf.predict([[1, 20]]))
print(clf.predict([[6, 10]]))
print(clf.predict([[3, 22]]))
*/

describe('naive_bayes:GaussianNB', () => {
  const X1 = [[1, 20], [20, 210], [3, 22], [40, 220], [6, 10], [7, 11]];
  const y1 = [1, 0, 1, 0, 2, 2];
  it('Should fit X1 and y1', () => {
    const nb:IMlModel<number> = new GaussianNB();
    nb.fit({ X: X1, y: y1 });
  });
  it('Should fit X1 and y1 then predict', () => {
    const expected = [1];
    const nb = new GaussianNB();
    nb.fit({ X: X1, y: y1 });
    const result = nb.predict({ X: [[1, 20]] });
    expect(result).toEqual(expected);
    const result2 = nb.predict({ X: [[25, 225]] });
    expect(result2).toEqual([0]);
    const result3 = nb.predict({ X: [[30, 215]] });
    expect(result3).toEqual([0]);
    const result4 = nb.predict({ X: [[3, 22]] });
    expect(result4).toEqual([1]);
    const result5 = nb.predict({ X: [[6, 10]] });
    expect(result5).toEqual([2]);
  });
  it('Should fit num and predict', () => {
    const nb = new GaussianNB();
    nb.fit({ X: X1, y: y1});
    const result = nb.predict({ X: [[1, 22]] });
    expect(result).toEqual([1]);
  });
  it('Should fit string and predict', () => {
    const nb = new GaussianNB<string>();
    nb.fit({ X: X1, y: y1.map(String) });
    const result = nb.predict({ X: [[25, 225]] });
    expect(result).toEqual(['0']);
  });
  it('Should fit X1 and y1 and reload then predict the same', () => {
    const expected = [1];

    // Initial model
    const nb = new GaussianNB();
    nb.fit({ X: X1, y: y1 });
    const result = nb.predict({ X: [[1, 20]] });
    expect(result).toEqual(expected);

    // Restored model
    const checkpoint = JSON.parse(JSON.stringify(nb.toJSON()));
    const nb2 = new GaussianNB();
    nb2.fromJSON(checkpoint);
    const result2 = nb2.predict({ X: [[1, 20]] });
    expect(result2).toEqual(expected);
  });
  it('Should not fit non array for training data', () => {
    const nb = new GaussianNB();
    const invalidMatrixMsg = 'X must be a matrix';
    expect(() => nb.fit({ X: 123, y: y1 })).toThrow(invalidMatrixMsg);
    expect(() => nb.fit({ X: [1, 2, 3], y: [1, 2] })).toThrow(invalidMatrixMsg);
    expect(() => nb.fit({ X: null, y: [1, 2] })).toThrow(invalidMatrixMsg);
  });
  it('Should not fit non array for testing data', () => {
    const nb = new GaussianNB();
    const invalidMatrixMsg = 'y must be a vector';
    const sizeNotEqual = 'X and y must be same in length';
    expect(() => nb.fit({ X: X1, y: 123 })).toThrow(invalidMatrixMsg);
    expect(() => nb.fit({ X: X1, y: null })).toThrow(invalidMatrixMsg);
    expect(() => nb.fit({ X: X1, y: [] })).toThrow(sizeNotEqual);
  });
  it('Should fit only accept X and y if number of attributes is same', () => {
    const nb = new GaussianNB();
    const sizeNotEqual = 'X and y must be same in length';
    expect(() => nb.fit({ X: X1, y: [1, 2, 3] })).toThrow(sizeNotEqual);
    expect(() => nb.fit({ X: [[1, 20], [2, 21], [3, 22]], y: y1 })).toThrow(sizeNotEqual);
  });
  it('should predict only accept X as matrix', () => {
    const nb = new GaussianNB();
    nb.fit({ X: X1, y: y1 });
    const invalidMatrixMsg = 'X must be a matrix';
    expect(() => nb.predict({ X: 1 })).toThrow(invalidMatrixMsg);
    expect(() => nb.predict({ X: null })).toThrow(invalidMatrixMsg);
    expect(() => nb.predict({ X: [1, 2, 3] })).toThrow(invalidMatrixMsg);
  });
  it('should not prediction attributes are greater than summary length', () => {
    const nb = new GaussianNB();
    nb.fit({ X: X1, y: y1 });
    const tooManyPredAttrs = 'Prediction input 4 length must be equal or less than summary length 2';
    expect(() => nb.predict({ X: [[1, 20, 11, 2]] })).toThrow(tooManyPredAttrs);
  });
});
