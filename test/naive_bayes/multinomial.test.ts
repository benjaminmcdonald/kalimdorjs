import { GaussianMN } from '../../src/lib/naive_bayes/multinomial';
/*
import numpy as np
X = np.array([[6, 9], [5, 5], [9, 5]])
y = np.array(['1', '2', '3'])
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X, y)

print(clf.predict(X[2:3]))
*/

describe('naive_bayes:GaussianMN', () => {
  const X1 = [[6, 9], [5, 5], [9, 5]];
  const y1 = ['1', '2', '3'];
/*
multinomial model
[
      0.40000012516975403,
      0.5999998450279236,
      0.5,
      0.5,
      0.6428568959236145,
      0.3571430742740631 ]
*/
  const expectedTests:ReadonlyArray<[[number, number], string]> = [
    [[6, 9], '1'],
    [[5, 5], '2'],
    [[9, 5], '3'],
    [[1, 9], '1'],
    [[9, 9], '2'],
    [[90, 9], '3'],
    [[1, 90], '1'],
    // [[10, 9], '2'],
    // [[10, -9], '3'],
    // [[1, 9], '3'],
  ];
  it('Should fit X1 and y1', () => {
    const nb = new GaussianMN<string>();
    nb.fit({X: X1, y: y1});

    expectedTests.forEach((expectedTest) => {
      const [test, expected] = expectedTest;
      const result = nb.predict([test]);
      expect(result).toEqual([expected]);
    });

    // test input as tests on fited model
    X1.forEach((test, i) => {
      const result = nb.predict([test]);
      expect(result).toEqual([y1[i]]);
    });
  });
  it('Should fit X1 and y1 as number', () => {
    const nb = new GaussianMN();
    nb.fit({X: X1, y: y1.map(y => +y)});

    expectedTests.forEach((expectedTest) => {
      const [test, expected] = expectedTest;
      const result = nb.predict([test]);
      expect(result).toEqual([+expected]);
    });
    X1.forEach((test, i) => {
      const result = nb.predict([test]);
      expect(result).toEqual([+y1[i]]);
    });
  });
  it('Should refit X1 and then predict the same', () => {
    const expected = ['1'];

    // Initial model
    const nb = new GaussianMN<string>();
    nb.fit({ X: X1, y: y1 });
    const result = nb.predict([[1, 20]]);
    expect(result).toEqual(expected);
    
    nb.fit({ X: X1, y: y1 });
    expect(nb.predict([[1, 20]])).toEqual(expected);
  });
  it('Should fit X1 and y1 and reload then predict the same', () => {
    const expected = ['1'];

    // Initial model
    const nb = new GaussianMN<string>();
    nb.fit({ X: X1, y: y1 });
    const result = nb.predict([[1, 20]]);
    expect(result).toEqual(expected);

    // Restored model
    const checkpoint = nb.toJSON();
    const nb2 = new GaussianMN<string>();
    nb2.fromJSON(checkpoint);
    const result2 = nb2.predict([[1, 20]]);
    expect(result2).toEqual(expected);
  });
});
