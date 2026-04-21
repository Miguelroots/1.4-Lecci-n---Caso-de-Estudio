# Importes para PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, countDistinct, when, sum as spark_sum,
    avg, stddev, max as spark_max, min as spark_min,
    mean, sqrt, round as spark_round
)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.mllib.clustering import KMeans as KMeans_RDD, GaussianMixture as GMM_RDD
# from pyspark.mllib.evaluation import ClusteringEvaluator as ClusteringEvaluator_RDD

# Importes para análisis de datos
import pandas as pd
import numpy as np

# Importes para visualización
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Configuración de visualización
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Suprimir advertencias
import warnings
warnings.filterwarnings('ignore')

print("✓ Librerías importadas exitosamente")
# Inicializar sesión Spark
spark = SparkSession.builder \
    .appName("Clustering_Logistica") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Spark Session Inicializada")
print(f"Versión: {spark.version}")
print(f"Aplicación: {spark.sparkContext.appName}")
# Rutas de datos
ruta_producto = r"producto_tabla/producto_tabla.csv"
ruta_test = r"test/test.csv"

# Leer tabla de productos
df_productos = spark.read.csv(ruta_producto, header=True, inferSchema=True)

# Leer datos de transacciones
df_transacciones = spark.read.csv(ruta_test, header=True, inferSchema=True)

print("=== TABLA DE PRODUCTOS ===")
print(f"Registros: {df_productos.count()}")
print(f"Columnas: {df_productos.columns}")
df_productos.show(5)

print("\n=== TABLA DE TRANSACCIONES ===")
print(f"Registros: {df_transacciones.count()}")
print(f"Columnas: {df_transacciones.columns}")
df_transacciones.show(5)
# Validar esquemas
print("ESQUEMA - Tabla de Productos:")
df_productos.printSchema()

print("\nESQUEMA - Tabla de Transacciones:")
df_transacciones.printSchema()
# Paso 3.1: Limpieza de tabla de productos
df_productos_clean = df_productos.filter(col('Producto_ID').isNotNull())
registros_eliminados_prod = df_productos.count() - df_productos_clean.count()

print(f"Tabla de Productos:")
print(f"  - Registros originales: {df_productos.count()}")
print(f"  - Registros con Producto_ID nulo eliminados: {registros_eliminados_prod}")
print(f"  - Registros finales: {df_productos_clean.count()}")
# Paso 3.2: Limpieza de tabla de transacciones
df_trans_clean = df_transacciones.filter(col('Producto_ID').isNotNull())
registros_eliminados_trans = df_transacciones.count() - df_trans_clean.count()

# Verificar valores en Semana
df_trans_clean.select('Semana').distinct().show()

print(f"\nTabla de Transacciones:")
print(f"  - Registros originales: {df_transacciones.count()}")
print(f"  - Registros con Producto_ID nulo eliminados: {registros_eliminados_trans}")
print(f"  - Registros finales: {df_trans_clean.count()}")
# Paso 3.3: Eliminar duplicados
df_trans_dedup = df_trans_clean.dropDuplicates()
duplicados_eliminados = df_trans_clean.count() - df_trans_dedup.count()

print(f"Duplicados eliminados en transacciones: {duplicados_eliminados}")
print(f"Registros después de deduplicación: {df_trans_dedup.count()}")
# Paso 4.1: Crear features por producto
# Feature 1: Volumen Total (número de transacciones por producto)
# Feature 2: Frecuencia de Pedidos (número de pedidos distintos)
# Feature 3: Concentración de Ventas (concentración en clientes)

features_df = df_trans_dedup.groupBy('Producto_ID').agg(
    # Feature 1: Volumen Total
    count('*').alias('Volumen_Total'),
    
    # Feature 2: Frecuencia de Pedidos (count de transacciones únicas)
    countDistinct('id').alias('Frecuencia_Pedidos'),
    
    # Feature 3: Concentración de Ventas (número de clientes únicos que compran el producto)
    countDistinct('Cliente_ID').alias('Clientes_Unicos')
)

# Calcular concentración como ratio entre transacciones y clientes
features_df = features_df.withColumn(
    'Concentracion_Ventas',
    spark_round(col('Volumen_Total') / col('Clientes_Unicos'), 2)
)

print("Features creadas por producto:")
features_df.show(10)
# Paso 4.2: Unificar con tabla de productos (JOIN)
df_datos_clustering = features_df.join(
    df_productos_clean,
    on='Producto_ID',
    how='inner'
)

print(f"Registros después del JOIN:")
print(f"  - Productos con datos de transacciones: {df_datos_clustering.count()}")

# Reordenar columnas
df_datos_clustering = df_datos_clustering.select(
    'Producto_ID',
    'NombreProducto',
    'Volumen_Total',
    'Frecuencia_Pedidos',
    'Clientes_Unicos',
    'Concentracion_Ventas'
)

print("\nDataFrame final para clustering:")
df_datos_clustering.show(10)
# Validar ausencia de valores nulos
print("Valores nulos por columna:")
for col_name in df_datos_clustering.columns:
    null_count = df_datos_clustering.filter(col(col_name).isNull()).count()
    print(f"  {col_name}: {null_count}")

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
df_datos_clustering.describe(['Volumen_Total', 'Frecuencia_Pedidos', 'Clientes_Unicos', 'Concentracion_Ventas']).show()
# Convertir a Pandas para visualizaciones
df_pandas = df_datos_clustering.select(
    'Producto_ID', 'Volumen_Total', 'Frecuencia_Pedidos', 'Clientes_Unicos', 'Concentracion_Ventas'
).toPandas()

print(f"DataFrame convertido a Pandas: {df_pandas.shape}")
print("\nPrimeras filas:")
print(df_pandas.head())
# EDA: Visualización de distribuciones
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribución de Features para Clustering', fontsize=14, fontweight='bold')

# Histograma Volumen Total
axes[0, 0].hist(df_pandas['Volumen_Total'], bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_xlabel('Volumen Total')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Volumen Total')
axes[0, 0].grid(alpha=0.3)

# Histograma Frecuencia de Pedidos
axes[0, 1].hist(df_pandas['Frecuencia_Pedidos'], bins=50, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('Frecuencia de Pedidos')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title('Distribución de Frecuencia de Pedidos')
axes[0, 1].grid(alpha=0.3)

# Histograma Clientes Únicos
axes[1, 0].hist(df_pandas['Clientes_Unicos'], bins=50, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Clientes Únicos')
axes[1, 0].set_ylabel('Frecuencia')
axes[1, 0].set_title('Distribución de Clientes Únicos')
axes[1, 0].grid(alpha=0.3)

# Histograma Concentración de Ventas
axes[1, 1].hist(df_pandas['Concentracion_Ventas'], bins=50, color='salmon', edgecolor='black')
axes[1, 1].set_xlabel('Concentración de Ventas')
axes[1, 1].set_ylabel('Frecuencia')
axes[1, 1].set_title('Distribución de Concentración de Ventas')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Histogramas de distribuciones generados")
# Scatter plots: Relaciones entre variables
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Relaciones entre Features', fontsize=14, fontweight='bold')

# Volumen vs Frecuencia
axes[0].scatter(df_pandas['Volumen_Total'], df_pandas['Frecuencia_Pedidos'], 
               alpha=0.5, s=50, color='steelblue')
axes[0].set_xlabel('Volumen Total')
axes[0].set_ylabel('Frecuencia de Pedidos')
axes[0].set_title('Volumen Total vs Frecuencia de Pedidos')
axes[0].grid(alpha=0.3)

# Frecuencia vs Concentración
axes[1].scatter(df_pandas['Frecuencia_Pedidos'], df_pandas['Concentracion_Ventas'],
               alpha=0.5, s=50, color='coral')
axes[1].set_xlabel('Frecuencia de Pedidos')
axes[1].set_ylabel('Concentración de Ventas')
axes[1].set_title('Frecuencia de Pedidos vs Concentración de Ventas')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Scatter plots generados")
# Matriz de correlación
correlation_matrix = df_pandas[['Volumen_Total', 'Frecuencia_Pedidos', 
                                  'Clientes_Unicos', 'Concentracion_Ventas']].corr()

print("Matriz de Correlación:")
print(correlation_matrix)

# Visualizar matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - Features de Clustering')
plt.tight_layout()
plt.show()

print("\n✓ Matriz de correlación visualizada")
# Estadísticas descriptivas completas
print("=== ESTADÍSTICAS DESCRIPTIVAS ===")
print("\nVolumen Total:")
print(f"  Media: {df_pandas['Volumen_Total'].mean():.2f}")
print(f"  Desv. Est.: {df_pandas['Volumen_Total'].std():.2f}")
print(f"  Min: {df_pandas['Volumen_Total'].min():.2f}")
print(f"  Max: {df_pandas['Volumen_Total'].max():.2f}")
print(f"  Mediana: {df_pandas['Volumen_Total'].median():.2f}")

print("\nFrecuencia de Pedidos:")
print(f"  Media: {df_pandas['Frecuencia_Pedidos'].mean():.2f}")
print(f"  Desv. Est.: {df_pandas['Frecuencia_Pedidos'].std():.2f}")
print(f"  Min: {df_pandas['Frecuencia_Pedidos'].min():.2f}")
print(f"  Max: {df_pandas['Frecuencia_Pedidos'].max():.2f}")
print(f"  Mediana: {df_pandas['Frecuencia_Pedidos'].median():.2f}")

print("\nClientes Únicos:")
print(f"  Media: {df_pandas['Clientes_Unicos'].mean():.2f}")
print(f"  Desv. Est.: {df_pandas['Clientes_Unicos'].std():.2f}")
print(f"  Min: {df_pandas['Clientes_Unicos'].min():.2f}")
print(f"  Max: {df_pandas['Clientes_Unicos'].max():.2f}")
print(f"  Mediana: {df_pandas['Clientes_Unicos'].median():.2f}")

print("\nConcentración de Ventas:")
print(f"  Media: {df_pandas['Concentracion_Ventas'].mean():.2f}")
print(f"  Desv. Est.: {df_pandas['Concentracion_Ventas'].std():.2f}")
print(f"  Min: {df_pandas['Concentracion_Ventas'].min():.2f}")
print(f"  Max: {df_pandas['Concentracion_Ventas'].max():.2f}")
print(f"  Mediana: {df_pandas['Concentracion_Ventas'].median():.2f}")
# Seleccionar features para clustering
features_cols = ['Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']

# Crear vector de features
assembler = VectorAssembler(inputCols=features_cols, outputCol='features')
df_features = assembler.transform(df_datos_clustering)

print("Features vectorizados:")
df_features.select('features').show(5)
# Normalizar features con StandardScaler
scaler = StandardScaler(inputCol='features', outputCol='features_scaled', withMean=True, withStd=True)
scaler_model = scaler.fit(df_features)
df_scaled = scaler_model.transform(df_features)

print("Features normalizados:")
df_scaled.select('features', 'features_scaled').show(5)

print("\n✓ Datos normalizados exitosamente")
# Determinar K óptimo usando Elbow Method y Silhouette Score
k_values = range(2, 11)
inertias = []
silhouette_scores_kmeans = []

evaluator = ClusteringEvaluator(predictionCol='prediction', 
                                featuresCol='features_scaled',
                                metricName='silhouette')

for k in k_values:
    kmeans = KMeans(k=k, seed=42, maxIter=100, featuresCol='features_scaled')
    model = kmeans.fit(df_scaled)
    
    # Predicciones
    predictions = model.transform(df_scaled)
    
    # Inercia (suma de distancias al cuadrado)
    inertia = model.computeCost(predictions)
    inertias.append(inertia)
    
    # Silhouette Score
    silhouette = evaluator.evaluate(predictions)
    silhouette_scores_kmeans.append(silhouette)
    
    print(f"K={k} | Inércia: {inertia:.2f} | Silhouette Score: {silhouette:.4f}")
# Visualizar Elbow Method y Silhouette Score
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Determinación de K Óptimo para K-means', fontsize=14, fontweight='bold')

# Elbow Method
axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inércia')
axes[0].set_title('Elbow Method')
axes[0].grid(alpha=0.3)

# Silhouette Score
axes[1].plot(k_values, silhouette_scores_kmeans, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por K')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Gráficos de determinación de K generados")
# Seleccionar K óptimo basado en Silhouette Score máximo
k_optimal = k_values[np.argmax(silhouette_scores_kmeans)]
max_silhouette = max(silhouette_scores_kmeans)

print(f"\n=== K ÓPTIMO SELECCIONADO ===")
print(f"K óptimo: {k_optimal}")
print(f"Silhouette Score máximo: {max_silhouette:.4f}")
print(f"Inércia asociada: {inertias[k_optimal - 2]:.2f}")
print(f"\nJustificación: Se selecciona K={k_optimal} por tener el Silhouette Score más alto ({max_silhouette:.4f}),")
print(f"indicando cohesión óptima de los clusters.")
# Entrenar K-means con K óptimo
kmeans_final = KMeans(k=k_optimal, seed=42, maxIter=100, featuresCol='features_scaled')
kmeans_model = kmeans_final.fit(df_scaled)

# Predicciones
df_kmeans = kmeans_model.transform(df_scaled)

# Evaluación
silhouette_kmeans_final = evaluator.evaluate(df_kmeans)
inertia_kmeans_final = kmeans_model.computeCost(df_kmeans)

print(f"=== MODELO K-MEANS FINAL ===")
print(f"K: {k_optimal}")
print(f"Silhouette Score: {silhouette_kmeans_final:.4f}")
print(f"Inércia: {inertia_kmeans_final:.2f}")
print(f"\nCentros de clusters (primeras 3 dimensiones):")
for i, center in enumerate(kmeans_model.clusterCenters()):
    print(f"  Cluster {i}: {center[:3]}")
# Agregar predicciones de K-means al dataframe
df_kmeans_results = df_kmeans.select('Producto_ID', 'NombreProducto', 
                                      'Volumen_Total', 'Frecuencia_Pedidos', 
                                      'Concentracion_Ventas', 'prediction')
df_kmeans_results = df_kmeans_results.withColumnRenamed('prediction', 'Cluster_KMeans')

print("Distribución de productos por cluster (K-means):")
df_kmeans_results.groupBy('Cluster_KMeans').count().show()
# Determinar número óptimo de componentes para GMM
n_components_values = range(2, 11)
bic_scores = []
aic_scores = []
silhouette_scores_gmm = []

for n_comp in n_components_values:
    gmm = GaussianMixture(k=n_comp, seed=42, maxIter=100, featuresCol='features_scaled')
    gmm_model = gmm.fit(df_scaled)
    
    # Predicciones
    predictions_gmm = gmm_model.transform(df_scaled)
    
    # BIC (Bayesian Information Criterion)
    bic = gmm_model.summary.bic
    bic_scores.append(bic)
    
    # Silhouette Score
    silhouette_gmm = evaluator.evaluate(predictions_gmm)
    silhouette_scores_gmm.append(silhouette_gmm)
    
    print(f"Componentes={n_comp} | BIC: {bic:.2f} | Silhouette Score: {silhouette_gmm:.4f}")
# Visualizar BIC y Silhouette Score para GMM
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Determinación de Componentes Óptimo para GMM', fontsize=14, fontweight='bold')

# BIC
axes[0].plot(n_components_values, bic_scores, 'go-', linewidth=2, markersize=8)
axes[0].set_xlabel('Número de Componentes')
axes[0].set_ylabel('BIC')
axes[0].set_title('BIC vs Componentes')
axes[0].grid(alpha=0.3)

# Silhouette Score
axes[1].plot(n_components_values, silhouette_scores_gmm, 'mo-', linewidth=2, markersize=8)
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score por Componentes')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Gráficos de GMM generados")
# Seleccionar número de componentes óptimo basado en BIC mínimo
n_components_optimal = n_components_values[np.argmin(bic_scores)]
min_bic = min(bic_scores)

print(f"\n=== COMPONENTES ÓPTIMO PARA GMM ===")
print(f"Componentes óptimo: {n_components_optimal}")
print(f"BIC mínimo: {min_bic:.2f}")
print(f"Silhouette Score asociado: {silhouette_scores_gmm[n_components_optimal - 2]:.4f}")
print(f"\nJustificación: Se selecciona {n_components_optimal} componentes por tener el BIC más bajo ({min_bic:.2f}),")
print(f"que indica mejor balance entre ajuste y complejidad del modelo.")
# Entrenar GMM con componentes óptimo
gmm_final = GaussianMixture(k=n_components_optimal, seed=42, maxIter=100, featuresCol='features_scaled')
gmm_model = gmm_final.fit(df_scaled)

# Predicciones
df_gmm = gmm_model.transform(df_scaled)

# Evaluación
silhouette_gmm_final = evaluator.evaluate(df_gmm)
bic_gmm_final = gmm_model.summary.bic

print(f"=== MODELO GMM FINAL ===")
print(f"Componentes: {n_components_optimal}")
print(f"Silhouette Score: {silhouette_gmm_final:.4f}")
print(f"BIC: {bic_gmm_final:.2f}")
# Agregar predicciones de GMM al dataframe
df_gmm_results = df_gmm.select('Producto_ID', 'NombreProducto', 
                                'Volumen_Total', 'Frecuencia_Pedidos', 
                                'Concentracion_Ventas', 'prediction')
df_gmm_results = df_gmm_results.withColumnRenamed('prediction', 'Cluster_GMM')

print("Distribución de productos por cluster (GMM):")
df_gmm_results.groupBy('Cluster_GMM').count().show()
# Tabla comparativa de métricas
print("\n=== COMPARACIÓN DE MODELOS ===")
print(f"\nK-MEANS:")
print(f"  - Número de clusters: {k_optimal}")
print(f"  - Silhouette Score: {silhouette_kmeans_final:.4f}")
print(f"  - Inércia: {inertia_kmeans_final:.2f}")

print(f"\nGAUSSIAN MIXTURE MODEL:")
print(f"  - Número de componentes: {n_components_optimal}")
print(f"  - Silhouette Score: {silhouette_gmm_final:.4f}")
print(f"  - BIC: {bic_gmm_final:.2f}")

print("\n=== ANÁLISIS CRÍTICO ===")
if silhouette_kmeans_final > silhouette_gmm_final:
    print(f"✓ K-means presenta mejor Silhouette Score ({silhouette_kmeans_final:.4f} vs {silhouette_gmm_final:.4f})")
    print("  Mayor cohesión interna de clusters.")
    modelo_recomendado = "K-means"
else:
    print(f"✓ GMM presenta mejor Silhouette Score ({silhouette_gmm_final:.4f} vs {silhouette_kmeans_final:.4f})")
    print("  Mayor cohesión interna de clusters.")
    modelo_recomendado = "GMM"

print(f"\n✓ MODELO RECOMENDADO: {modelo_recomendado}")
print(f"  Justificación: Mayor Silhouette Score y mejor interpretabilidad para gestión de inventario.")
# Unificar resultados de ambos modelos para análisis
df_todos_clusters = df_kmeans_results.join(
    df_gmm_results.select('Producto_ID', 'Cluster_GMM'),
    on='Producto_ID'
)

# Convertir a Pandas para análisis
df_clusters_pandas = df_todos_clusters.toPandas()

print(f"Dataframe con clusters unificados: {df_clusters_pandas.shape}")
print("\nPrimeras filas:")
print(df_clusters_pandas.head())
# Caracterización de clusters K-means
print("\n" + "="*60)
print("CARACTERIZACIÓN DE CLUSTERS - K-MEANS (K={})".format(k_optimal))
print("="*60)

cluster_names = {}  # Diccionario para almacenar nombres descriptivos

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    
    n_products = len(cluster_data)
    vol_mean = cluster_data['Volumen_Total'].mean()
    vol_min = cluster_data['Volumen_Total'].min()
    vol_max = cluster_data['Volumen_Total'].max()
    
    freq_mean = cluster_data['Frecuencia_Pedidos'].mean()
    freq_min = cluster_data['Frecuencia_Pedidos'].min()
    freq_max = cluster_data['Frecuencia_Pedidos'].max()
    
    conc_mean = cluster_data['Concentracion_Ventas'].mean()
    conc_min = cluster_data['Concentracion_Ventas'].min()
    conc_max = cluster_data['Concentracion_Ventas'].max()
    
    print(f"\n{'─'*50}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'─'*50}")
    print(f"Cantidad de productos: {n_products}")
    
    print(f"\nVolumen Total:")
    print(f"  Rango: {vol_min:.0f} - {vol_max:.0f}")
    print(f"  Promedio: {vol_mean:.2f}")
    
    print(f"\nFrecuencia de Pedidos:")
    print(f"  Rango: {freq_min:.0f} - {freq_max:.0f}")
    print(f"  Promedio: {freq_mean:.2f}")
    
    print(f"\nConcentración de Ventas:")
    print(f"  Rango: {conc_min:.2f} - {conc_max:.2f}")
    print(f"  Promedio: {conc_mean:.2f}")
    
    # Asignar nombre descriptivo basado en características
    if vol_mean > df_clusters_pandas['Volumen_Total'].median() and freq_mean > df_clusters_pandas['Frecuencia_Pedidos'].median():
        if conc_mean > df_clusters_pandas['Concentracion_Ventas'].median():
            cluster_name = "Alta Rotación Centralizada"
        else:
            cluster_name = "Alta Rotación Dispersa"
    elif vol_mean < df_clusters_pandas['Volumen_Total'].median() and freq_mean < df_clusters_pandas['Frecuencia_Pedidos'].median():
        cluster_name = "Baja Rotación"
    else:
        cluster_name = "Rotación Media"
    
    cluster_names[cluster_id] = cluster_name
    print(f"\n✓ Nombre descriptivo: {cluster_name}")
    
    # Top 5 productos en el cluster
    top_products = cluster_data.nlargest(5, 'Volumen_Total')[['Producto_ID', 'NombreProducto', 'Volumen_Total', 'Frecuencia_Pedidos']]
    print(f"\nTop 5 productos (por volumen):")
    for idx, row in top_products.iterrows():
        print(f"  - {row['NombreProducto']} (ID: {row['Producto_ID']}) - Vol: {row['Volumen_Total']:.0f}, Freq: {row['Frecuencia_Pedidos']:.0f}")
# Tabla resumen de clusters
print("\n" + "="*80)
print("TABLA RESUMEN - CLUSTERS K-MEANS")
print("="*80)

resumen_clusters = []

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    
    resumen_clusters.append({
        'Cluster': cluster_id,
        'Nombre': cluster_names.get(cluster_id, 'Sin definir'),
        'Productos': len(cluster_data),
        'Vol Promedio': f"{cluster_data['Volumen_Total'].mean():.1f}",
        'Freq Promedio': f"{cluster_data['Frecuencia_Pedidos'].mean():.1f}",
        'Conc Promedio': f"{cluster_data['Concentracion_Ventas'].mean():.2f}"
    })

df_resumen = pd.DataFrame(resumen_clusters)
print(df_resumen.to_string(index=False))
# Definir colores para clusters
colores = plt.cm.Set3(np.linspace(0, 1, k_optimal))

# Visualización 2D: Volumen vs Frecuencia coloreado por cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Visualización de Clusters - K-means', fontsize=14, fontweight='bold')

# Scatter Volumen vs Frecuencia
for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    axes[0].scatter(cluster_data['Volumen_Total'], 
                   cluster_data['Frecuencia_Pedidos'],
                   c=[colores[cluster_id]], 
                   label=f'Cluster {cluster_id}: {cluster_names.get(cluster_id, "Sin definir")}',
                   s=100, alpha=0.6, edgecolors='black')

axes[0].set_xlabel('Volumen Total (número de transacciones)', fontsize=11)
axes[0].set_ylabel('Frecuencia de Pedidos', fontsize=11)
axes[0].set_title('Volumen vs Frecuencia de Pedidos', fontsize=12, fontweight='bold')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(alpha=0.3)

# Scatter Frecuencia vs Concentración
for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    axes[1].scatter(cluster_data['Frecuencia_Pedidos'],
                   cluster_data['Concentracion_Ventas'],
                   c=[colores[cluster_id]],
                   label=f'Cluster {cluster_id}: {cluster_names.get(cluster_id, "Sin definir")}',
                   s=100, alpha=0.6, edgecolors='black')

axes[1].set_xlabel('Frecuencia de Pedidos', fontsize=11)
axes[1].set_ylabel('Concentración de Ventas', fontsize=11)
axes[1].set_title('Frecuencia vs Concentración de Ventas', fontsize=12, fontweight='bold')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualización 2D generada")
# Box plots de features por cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribución de Features por Cluster - K-means', fontsize=14, fontweight='bold')

# Box plot Volumen Total
df_clusters_pandas.boxplot(column='Volumen_Total', by='Cluster_KMeans', ax=axes[0, 0])
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Volumen Total')
axes[0, 0].set_title('Volumen Total por Cluster')
axes[0, 0].get_figure().suptitle('')  # Remover título duplicado

# Box plot Frecuencia
df_clusters_pandas.boxplot(column='Frecuencia_Pedidos', by='Cluster_KMeans', ax=axes[0, 1])
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Frecuencia de Pedidos')
axes[0, 1].set_title('Frecuencia de Pedidos por Cluster')
axes[0, 1].get_figure().suptitle('')

# Box plot Concentración
df_clusters_pandas.boxplot(column='Concentracion_Ventas', by='Cluster_KMeans', ax=axes[1, 0])
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Concentración de Ventas')
axes[1, 0].set_title('Concentración de Ventas por Cluster')
axes[1, 0].get_figure().suptitle('')

# Distribución de productos por cluster
cluster_counts = df_clusters_pandas['Cluster_KMeans'].value_counts().sort_index()
axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color=colores, edgecolor='black')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Cantidad de Productos')
axes[1, 1].set_title('Distribución de Productos por Cluster')
axes[1, 1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("✓ Box plots generados")
# Visualización 3D: Volumen vs Frecuencia vs Concentración
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    ax.scatter(cluster_data['Volumen_Total'],
              cluster_data['Frecuencia_Pedidos'],
              cluster_data['Concentracion_Ventas'],
              c=[colores[cluster_id]],
              label=f'Cluster {cluster_id}: {cluster_names.get(cluster_id, "Sin definir")}',
              s=100, alpha=0.6, edgecolors='black')

ax.set_xlabel('Volumen Total', fontsize=11, fontweight='bold')
ax.set_ylabel('Frecuencia de Pedidos', fontsize=11, fontweight='bold')
ax.set_zlabel('Concentración de Ventas', fontsize=11, fontweight='bold')
ax.set_title('Visualización 3D de Clusters - K-means\n(Volumen vs Frecuencia vs Concentración)', 
             fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("✓ Visualización 3D generada")
print("\n" + "="*80)
print("ESTRATEGIAS DE GESTIÓN DE INVENTARIO POR CLUSTER")
print("="*80)

estrategias = {
    "Alta Rotación Centralizada": {
        "descripcion": "Productos muy solicitados, vendidos principalmente a pocos clientes grandes",
        "almacenamiento": "Zona de fácil acceso próxima a carga/descarga - Ubicación estratégica A1",
        "replenishment": "Frecuencia: Semanal o bi-semanal - Volúmenes: Grandes",
        "nivel_stock": "Alto (3-4 semanas de cobertura)",
        "rotacion": "Muy Alta (menos de 7 días por ciclo)",
        "kpis": "Tasa de ruptura < 2% | Tasa de obsolescencia < 1%"
    },
    "Alta Rotación Dispersa": {
        "descripcion": "Productos muy solicitados, vendidos a muchos clientes diferentes",
        "almacenamiento": "Zona de acceso fácil pero distribución frecuente - Ubicación estratégica A2",
        "replenishment": "Frecuencia: Semanal - Volúmenes: Medianos",
        "nivel_stock": "Medio-Alto (2-3 semanas de cobertura)",
        "rotacion": "Alta (7-14 días por ciclo)",
        "kpis": "Tasa de ruptura < 3% | Eficiencia pick/pack > 95%"
    },
    "Rotación Media": {
        "descripcion": "Productos con demanda moderada y consistente",
        "almacenamiento": "Zona media del almacén - Ubicación estratégica B",
        "replenishment": "Frecuencia: Bi-semanal - Volúmenes: Pequeños a medianos",
        "nivel_stock": "Medio (2 semanas de cobertura)",
        "rotacion": "Media (14-30 días por ciclo)",
        "kpis": "Tasa de ruptura < 5% | Tasa de cobertura > 90%"
    },
    "Baja Rotación": {
        "descripcion": "Productos especializados con demanda baja o esporádica",
        "almacenamiento": "Zona profunda del almacén - Ubicación estratégica C (fondo)",
        "replenishment": "Frecuencia: Mensual o bajo demanda - Volúmenes: Muy pequeños",
        "nivel_stock": "Bajo (4+ semanas de cobertura o bajo demanda)",
        "rotacion": "Baja (más de 30 días por ciclo)",
        "kpis": "Tasa de obsolescencia < 10% | Costo de almacenamiento minimizado"
    }
}

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    nombre_cluster = cluster_names.get(cluster_id, "Sin definir")
    estrategia = estrategias.get(nombre_cluster, {})
    
    print(f"\n{'─'*80}")
    print(f"CLUSTER {cluster_id}: {nombre_cluster.upper()}")
    print(f"{'─'*80}")
    
    if estrategia:
        for key, value in estrategia.items():
            print(f"\n{key.upper().replace('_', ' ')}:")
            print(f"  {value}")
    else:
        print("Cluster no mapeado a estrategia predefinida")
print("\n" + "="*80)
print("CONCLUSIONES DEL ANÁLISIS DE CLUSTERING")
print("="*80)

print(f"""
RESUMEN EJECUTIVO:
─────────────────

1. MODELO SELECCIONADO: K-means con K={k_optimal}
   - Silhouette Score: {silhouette_kmeans_final:.4f}
   - Inércia: {inertia_kmeans_final:.2f}
   - Razón: Mayor cohesión y mejor interpretabilidad para el negocio

2. CLUSTERS IDENTIFICADOS: {k_optimal} segmentos de productos
""")

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    nombre = cluster_names.get(cluster_id, "Sin definir")
    cantidad = len(df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id])
    print(f"   - Cluster {cluster_id}: '{nombre}' ({cantidad} productos)")

print(f"""
3. FEATURES UTILIZADAS (3 características normalizadas):
   ✓ Volumen Total: Cantidad total de transacciones por producto
   ✓ Frecuencia de Pedidos: Número de pedidos distintos
   ✓ Concentración de Ventas: Ratio entre transacciones y clientes únicos

4. BENEFICIOS ESPERADOS:
   ✓ Optimización de espacios físicos en almacén (zonas estratégicas A, B, C)
   ✓ Planificación mejorada de replenishment según cluster
   ✓ Reducción de costos operativos (almacenamiento, picking, packaging)
   ✓ Minimización de rupturas de stock para productos críticos
   ✓ Mejor previsión de demanda por segmento

5. LIMITACIONES DEL ESTUDIO:
   ⚠ Datos limitados a 2 semanas (semanas 10-11 del año)
   ⚠ No captura estacionalidad o picos de demanda
   ⚠ Suposición: Las 2 semanas son representativas del patrón normal
   ⚠ Variables externas no consideradas (precio, promociones, climatología)

6. RECOMENDACIONES PARA VALIDACIÓN EN PRODUCCIÓN:
   → Extender análisis a mínimo 12 semanas de datos históricos
   → Capturar períodos con diferentes patrones (temporada alta/baja)
   → Incluir variables adicionales: costo del producto, margen de ganancia
   → Validar con expertos de logística antes de implementar
   → Monitorear cambios en la demanda y re-clustering mensual
   → Implementar alertas automáticas para transiciones entre clusters

7. PRÓXIMOS PASOS:
   → Pilotar con 1-2 clusters antes de implementación completa
   → Medir impacto en: tiempos de picking, costos de storage, rotación
   → Ajustar estrategias según feedback operacional
   → Automatizar re-clustering en pipeline de datos
""")

print("="*80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE ✓")
print("="*80)
# Guardar resultados en archivo CSV para auditoría
df_clusters_pandas.to_csv(r'resultados_clustering.csv', index=False)
print("✓ Resultados guardados en: resultados_clustering.csv")
print(f"\n  Total de registros: {len(df_clusters_pandas)}")
print("  Columnas incluidas:")
for col in df_clusters_pandas.columns:
    print(f"    - {col}")
