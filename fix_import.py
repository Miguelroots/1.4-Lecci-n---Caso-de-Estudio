import io

with io.open('Clustering_Logistica.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('from pyspark.mllib.evaluation import ClusteringEvaluator as ClusteringEvaluator_RDD\\n', 
                          '# from pyspark.mllib.evaluation import ClusteringEvaluator as ClusteringEvaluator_RDD\\n')

with io.open('Clustering_Logistica.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)
