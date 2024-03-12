# Weekly Meeting

> Giovanni Foletto

To present:

Log deep usage and data model:

- the evaluation of the logdeep model in order to train the data
- data quality assumption in the data elaborated to inserted (see results 0, 2, 3, 4, 5)
- The results of the logDeep model are unexpected. Need to investigate more
- the correlation with `block` clearly does not work, searching for other methods (like IP/User)
    - using this two methods is not useful in this dataset, there are always the same IP and the same users
    - could be error in the anonimization or in the data itself (maybe the connection information are the same, and the anonimization software returned the data at this)


K-means/K-nearest-neighbours/DataSet

- The problems with classification is correct
- The graph highlight a simulation in which the majority of the data are between a limited space.
- The value is `594` (`RunInstances`) and maybe to be useful could be intersected with some other information (like `requestsObject`)
- The *scatterplot* initially elaborate this information that has revealed correct within the consideration of later exploratory analysis.
- On that data are elaborated 2 different K-nearest algorithm (DBSCAN, Kmeans) taken from ScikitLearn. The centroid are really few relatively to the expected results (at least half of the EventName types).
- The last graph on 09_clustering represent the problem good.

Bert:

- The bert classfication for the text is really good and allow a better classification algorithms that the bagging or heuristic methods presents
- https://arxiv.org/pdf/2307.09950.pdf => elaborate on classification with LLM, ad elaborate a request to be done withing ChatGPT API.
- this solution is WIP => tested only the pretrained solution withing COLAB 