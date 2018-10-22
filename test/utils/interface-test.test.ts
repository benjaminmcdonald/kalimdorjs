import { KMeans } from '../../src/lib/cluster/k_means';
import { RandomForestClassifier } from '../../src/lib/ensemble/forest';
import { LinearRegression } from '../../src/lib/linear_model/linear_regression';
import { IMlModel } from '../../src/lib/model-interfaces';
import { GaussianNB } from '../../src/lib/naive_bayes';
import { KNeighborsClassifier } from '../../src/lib/neighbors/classification';

describe('types:IMlModel', () => {
  it('Should allow IMlModel', () => {
    const model:IMlModel<number> = new GaussianNB();
    const model2:IMlModel<number> = new KMeans();
    const model3:IMlModel<number> = new RandomForestClassifier();
    const model4:IMlModel<number> = new LinearRegression();
    const model5:IMlModel<number> = new KNeighborsClassifier();
  });
});
