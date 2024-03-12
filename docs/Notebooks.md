# Notebooks.md

This file countains the all the explanation about the notebook that are not clear enought by themselves.

## 08_Log_From_LogPai_Drain

This notebook is create to initially tests and elaborate on the `logparser.Drain` that is a newest implementation of the `Drain` parser, rewritten by the `logpai` community in order to solve some problem in the original implemenation (`drain3` PyPl packet).

They elaborate on that allowing the:

- online working (the intial drain implementaion is only ondemand)
- other things, like saving data in a contextual and better format like the `csv`.

### Testing With different log template

Maybe useful:

The code that retuns `Warning Skip line` is this:

There are some solution that return less error than the previous:

1. Test with `'<Date> <Time> <Level>:<Content>'` and without a regex. This return only 21 results. There is a correct every 100.000 entry.
=> Errors and 1428 templates.

2. Test with `<Date> <Time>: <Content>` => No errors (500 Templates)

3. Test with `<Date> <Time>: <Content>` but the content ordered. Less performant (but I should have been expected). => No errors. 410 templates

4. Test like `2` but activate the regex for IP.

5. Test like `3` but activate the regex for the IP

TODO: cross-reference on information-loss and what is get lost.

TODO: data in the structured format MUST be improved.
The datetime manipulation need enhancement.
To test with:
```python
TIME_STRING_FORMAT = "%Y-%m-%dT%H:%M:%S:Z"
```
(note the other `:` added before `Z`)

TODO:

- better handling of the `eventID` codes
- better handling of complex data substructure
- 

## Evaluation with LogDeep Models

To evaluate with this model, we need to rewrite the sampling method in order to be readed as intended.

In the sampling method, with HDFS, they uses a regex-function in order to find all `block` section inside the logs and so correlate the information of each block within itself.

## 09_Clustering.ipynb

Using the clustering or K-means to represent the anomaly in the dataset.

Using the standard scaler we found a estimated number of clusters of 4/5. This could not be the case and it is probably wrong.

The dataset is heavily unbalanced:

- event number `594` (`RunInstances`) appear to have a 62% apparition rate. That means more than half of the dataset is this event.

- 



