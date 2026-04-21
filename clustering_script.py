#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering de Productos en PySpark
Caso de Estudio: Segmentación de Inventario en Logística
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, countDistinct, round as spark_round
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CLUSTERING DE PRODUCTOS EN PYSPARK")
print("Segmentación de Inventario en Logística")
print("="*80)

# ========== Configuración ==========
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# ========== Inicializar Spark ==========
spark = SparkSession.builder.appName("Clustering_Logistica") \
    .config("spark.sql.adaptive.enabled", "true").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print(f"\n✓ Spark Session iniciada: {spark.sparkContext.appName}")

# ========== Cargar Datos ==========
print("\n" + "="*80)
print("FASE 1: CARGA DE DATOS")
print("="*80)

ruta_producto = r"c:\Users\USER\Downloads\leccion\producto_tabla\producto_tabla.csv"
ruta_test = r"c:\Users\USER\Downloads\leccion\test\test.csv"

df_productos = spark.read.csv(ruta_producto, header=True, inferSchema=True)
df_transacciones = spark.read.csv(ruta_test, header=True, inferSchema=True)

print(f"\nTabla de Productos: {df_productos.count()} registros")
print(f"Columnas: {df_productos.columns[:3]}...")
df_productos.show(2)

print(f"\nTabla de Transacciones: {df_transacciones.count()} registros")
print(f"Columnas: {df_transacciones.columns}")
df_transacciones.show(2)

# ========== Preprocesamiento ==========
print("\n" + "="*80)
print("FASE 2: PREPROCESAMIENTO DE DATOS")
print("="*80)

df_productos_clean = df_productos.filter(col('Producto_ID').isNotNull())
df_trans_clean = df_transacciones.filter(col('Producto_ID').isNotNull()).dropDuplicates()

print(f"\nProductos válidos: {df_productos_clean.count()}")
print(f"Transacciones válidas: {df_trans_clean.count()}")

# ========== Feature Engineering ==========
print("\n" + "="*80)
print("FASE 3: FEATURE ENGINEERING")
print("="*80)

features_df = df_trans_clean.groupBy('Producto_ID').agg(
    count('*').alias('Volumen_Total'),
    countDistinct('id').alias('Frecuencia_Pedidos'),
    countDistinct('Cliente_ID').alias('Clientes_Unicos')
)

features_df = features_df.withColumn('Concentracion_Ventas',
    spark_round(col('Volumen_Total') / col('Clientes_Unicos'), 2))

df_clustering = features_df.join(df_productos_clean, on='Producto_ID', how='inner').select(
    'Producto_ID', 'NombreProducto', 'Volumen_Total', 
    'Frecuencia_Pedidos', 'Clientes_Unicos', 'Concentracion_Ventas')

print(f"\nProductos con features: {df_clustering.count()}")
print("\nEstadísticas descriptivas:")
df_clustering.describe(['Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']).show()

# ========== Conversión a Pandas ==========
df_pandas = df_clustering.select(
    'Producto_ID', 'Volumen_Total', 'Frecuencia_Pedidos', 
    'Clientes_Unicos', 'Concentracion_Ventas').toPandas()

print(f"\nDataFrame Pandas convertido: {df_pandas.shape[0]} productos")

# ========== EDA ==========
print("\n" + "="*80)
print("FASE 4: ANÁLISIS EXPLORATORIO (EDA)")
print("="*80)

print("\nGenerando visualizaciones de distribuciones...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribución de Features para Clustering', fontsize=14, fontweight='bold')

axes[0, 0].hist(df_pandas['Volumen_Total'], bins=50, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Volumen Total')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].hist(df_pandas['Frecuencia_Pedidos'], bins=50, color='coral', edgecolor='black')
axes[0, 1].set_title('Frecuencia de Pedidos')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].hist(df_pandas['Clientes_Unicos'], bins=50, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Clientes Únicos')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].hist(df_pandas['Concentracion_Ventas'], bins=50, color='salmon', edgecolor='black')
axes[1, 1].set_title('Concentración de Ventas')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\USER\Downloads\leccion\01_Distribuciones.png')
plt.show()

print("✓ Histogramas generados")

# ========== Normalización ==========
print("\n" + "="*80)
print("FASE 5: NORMALIZACIÓN DE DATOS")
print("="*80)

features_cols = ['Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas']
assembler = VectorAssembler(inputCols=features_cols, outputCol='features')
df_features = assembler.transform(df_clustering)

scaler = StandardScaler(inputCol='features', outputCol='features_scaled', 
                       withMean=True, withStd=True)
scaler_model = scaler.fit(df_features)
df_scaled = scaler_model.transform(df_features)

print("✓ Features normalizados con StandardScaler")

# ========== K-MEANS ==========
print("\n" + "="*80)
print("FASE 6: K-MEANS - DETERMINACIÓN DE K ÓPTIMO")
print("="*80)

k_values = range(2, 11)
inertias = []
silhouette_scores_kmeans = []

evaluator = ClusteringEvaluator(predictionCol='prediction', 
                                featuresCol='features_scaled', metricName='silhouette')

print("\nCalculando K óptimo...")
for k in k_values:
    kmeans = KMeans(k=k, seed=42, maxIter=100, featuresCol='features_scaled')
    model = kmeans.fit(df_scaled)
    predictions = model.transform(df_scaled)
    inertia = model.computeCost(predictions)
    silhouette = evaluator.evaluate(predictions)
    inertias.append(inertia)
    silhouette_scores_kmeans.append(silhouette)
    print(f"  K={k:2d} | Inércia: {inertia:10.2f} | Silhouette: {silhouette:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Determinación de K Óptimo para K-means', fontsize=14, fontweight='bold')

axes[0].plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('K')
axes[0].set_ylabel('Inércia')
axes[0].set_title('Elbow Method')
axes[0].grid(alpha=0.3)

axes[1].plot(k_values, silhouette_scores_kmeans, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('K')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\USER\Downloads\leccion\02_K_Optimo.png')
plt.show()

k_optimal = k_values[np.argmax(silhouette_scores_kmeans)]
print(f"\n✓ K ÓPTIMO SELECCIONADO: {k_optimal}")
print(f"  Silhouette Score máximo: {max(silhouette_scores_kmeans):.4f}")

# ========== Entrenamiento K-MEANS ==========
print("\n" + "="*80)
print("FASE 7: K-MEANS - ENTRENAMIENTO FINAL")
print("="*80)

kmeans_final = KMeans(k=k_optimal, seed=42, maxIter=100, featuresCol='features_scaled')
kmeans_model = kmeans_final.fit(df_scaled)
df_kmeans = kmeans_model.transform(df_scaled)

silhouette_kmeans_final = evaluator.evaluate(df_kmeans)
inertia_kmeans_final = kmeans_model.computeCost(df_kmeans)

print(f"\nModelo K-MEANS entrenado:")
print(f"  K: {k_optimal}")
print(f"  Silhouette Score: {silhouette_kmeans_final:.4f}")
print(f"  Inércia: {inertia_kmeans_final:.2f}")

df_kmeans_results = df_kmeans.select('Producto_ID', 'NombreProducto', 
    'Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas', 'prediction')
df_kmeans_results = df_kmeans_results.withColumnRenamed('prediction', 'Cluster_KMeans')

print("\nDistribución de productos por cluster:")
df_kmeans_results.groupBy('Cluster_KMeans').count().orderBy('Cluster_KMeans').show()

# ========== GMM ==========
print("\n" + "="*80)
print("FASE 8: GMM - DETERMINACIÓN DE COMPONENTES ÓPTIMO")
print("="*80)

n_components_values = range(2, 11)
bic_scores = []
silhouette_scores_gmm = []

print("\nCalculando componentes óptimo...")
for n_comp in n_components_values:
    gmm = GaussianMixture(k=n_comp, seed=42, maxIter=100, featuresCol='features_scaled')
    gmm_model = gmm.fit(df_scaled)
    predictions_gmm = gmm_model.transform(df_scaled)
    bic = gmm_model.summary.bic
    silhouette_gmm = evaluator.evaluate(predictions_gmm)
    bic_scores.append(bic)
    silhouette_scores_gmm.append(silhouette_gmm)
    print(f"  Comp={n_comp:2d} | BIC: {bic:10.2f} | Silhouette: {silhouette_gmm:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(n_components_values, bic_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Número de Componentes')
plt.ylabel('BIC')
plt.title('BIC vs Componentes - GMM')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(r'c:\Users\USER\Downloads\leccion\03_GMM_BIC.png')
plt.show()

n_components_optimal = n_components_values[np.argmin(bic_scores)]
print(f"\n✓ COMPONENTES ÓPTIMO SELECCIONADO: {n_components_optimal}")
print(f"  BIC mínimo: {min(bic_scores):.2f}")

# ========== Entrenamiento GMM ==========
print("\n" + "="*80)
print("FASE 9: GMM - ENTRENAMIENTO FINAL")
print("="*80)

gmm_final = GaussianMixture(k=n_components_optimal, seed=42, maxIter=100, featuresCol='features_scaled')
gmm_model = gmm_final.fit(df_scaled)
df_gmm = gmm_model.transform(df_scaled)

silhouette_gmm_final = evaluator.evaluate(df_gmm)
bic_gmm_final = gmm_model.summary.bic

print(f"\nModelo GMM entrenado:")
print(f"  Componentes: {n_components_optimal}")
print(f"  Silhouette Score: {silhouette_gmm_final:.4f}")
print(f"  BIC: {bic_gmm_final:.2f}")

df_gmm_results = df_gmm.select('Producto_ID', 'NombreProducto', 
    'Volumen_Total', 'Frecuencia_Pedidos', 'Concentracion_Ventas', 'prediction')
df_gmm_results = df_gmm_results.withColumnRenamed('prediction', 'Cluster_GMM')

print("\nDistribución de productos por cluster:")
df_gmm_results.groupBy('Cluster_GMM').count().orderBy('Cluster_GMM').show()

# ========== Comparación ==========
print("\n" + "="*80)
print("FASE 10: COMPARACIÓN DE MODELOS")
print("="*80)

print(f"\nK-MEANS:")
print(f"  Clusters: {k_optimal}")
print(f"  Silhouette Score: {silhouette_kmeans_final:.4f}")
print(f"  Inércia: {inertia_kmeans_final:.2f}")

print(f"\nGAUSSIAN MIXTURE MODEL:")
print(f"  Componentes: {n_components_optimal}")
print(f"  Silhouette Score: {silhouette_gmm_final:.4f}")
print(f"  BIC: {bic_gmm_final:.2f}")

if silhouette_kmeans_final > silhouette_gmm_final:
    print(f"\n✓ MODELO RECOMENDADO: K-MEANS")
    print(f"  Justificación: Mayor Silhouette Score ({silhouette_kmeans_final:.4f} vs {silhouette_gmm_final:.4f})")
else:
    print(f"\n✓ MODELO RECOMENDADO: GMM")
    print(f"  Justificación: Mayor Silhouette Score ({silhouette_gmm_final:.4f} vs {silhouette_kmeans_final:.4f})")

# ========== Unificar Resultados ==========
df_todos_clusters = df_kmeans_results.join(
    df_gmm_results.select('Producto_ID', 'Cluster_GMM'), on='Producto_ID')
df_clusters_pandas = df_todos_clusters.toPandas()

print(f"\n✓ Resultados unificados: {df_clusters_pandas.shape[0]} productos")

# ========== Interpretación ==========
print("\n" + "="*80)
print("FASE 11: INTERPRETACIÓN DE CLUSTERS")
print("="*80)

cluster_names = {}
for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    
    n_products = len(cluster_data)
    vol_mean = cluster_data['Volumen_Total'].mean()
    vol_min = cluster_data['Volumen_Total'].min()
    vol_max = cluster_data['Volumen_Total'].max()
    freq_mean = cluster_data['Frecuencia_Pedidos'].mean()
    conc_mean = cluster_data['Concentracion_Ventas'].mean()
    
    print(f"\n{'─'*70}")
    print(f"CLUSTER {cluster_id} ({n_products} productos)")
    print(f"{'─'*70}")
    print(f"  Volumen Total: {vol_min:.0f} - {vol_max:.0f} (promedio: {vol_mean:.2f})")
    print(f"  Frecuencia de Pedidos: {freq_mean:.2f}")
    print(f"  Concentración de Ventas: {conc_mean:.2f}")
    
    if vol_mean > df_clusters_pandas['Volumen_Total'].median() and \
       freq_mean > df_clusters_pandas['Frecuencia_Pedidos'].median():
        if conc_mean > df_clusters_pandas['Concentracion_Ventas'].median():
            cluster_name = "Alta Rotación Centralizada"
        else:
            cluster_name = "Alta Rotación Dispersa"
    elif vol_mean < df_clusters_pandas['Volumen_Total'].median() and \
         freq_mean < df_clusters_pandas['Frecuencia_Pedidos'].median():
        cluster_name = "Baja Rotación"
    else:
        cluster_name = "Rotación Media"
    
    cluster_names[cluster_id] = cluster_name
    print(f"\n  ✓ NOMBRE DESCRIPTIVO: {cluster_name}")
    
    top_products = cluster_data.nlargest(5, 'Volumen_Total')[['NombreProducto', 'Volumen_Total', 'Frecuencia_Pedidos']]
    print(f"\n  TOP 5 PRODUCTOS (por volumen):")
    for idx, row in top_products.iterrows():
        prod_name = row['NombreProducto'][:50]
        print(f"    • {prod_name} (Vol: {row['Volumen_Total']:.0f}, Freq: {row['Frecuencia_Pedidos']:.0f})")

# ========== Visualizaciones 2D ==========
print("\n" + "="*80)
print("FASE 12: VISUALIZACIONES 2D")
print("="*80)

colores = plt.cm.Set3(np.linspace(0, 1, k_optimal))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Visualización de Clusters - K-means (2D)', fontsize=14, fontweight='bold')

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    axes[0].scatter(cluster_data['Volumen_Total'], 
                   cluster_data['Frecuencia_Pedidos'],
                   c=[colores[cluster_id]], 
                   label=f"Cluster {cluster_id}: {cluster_names.get(cluster_id, 'Sin def')}",
                   s=100, alpha=0.6, edgecolors='black')

axes[0].set_xlabel('Volumen Total (número de transacciones)', fontsize=11)
axes[0].set_ylabel('Frecuencia de Pedidos', fontsize=11)
axes[0].set_title('Volumen Total vs Frecuencia de Pedidos')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(alpha=0.3)

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    axes[1].scatter(cluster_data['Frecuencia_Pedidos'],
                   cluster_data['Concentracion_Ventas'],
                   c=[colores[cluster_id]],
                   s=100, alpha=0.6, edgecolors='black')

axes[1].set_xlabel('Frecuencia de Pedidos', fontsize=11)
axes[1].set_ylabel('Concentración de Ventas', fontsize=11)
axes[1].set_title('Frecuencia vs Concentración de Ventas')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\USER\Downloads\leccion\04_Clusters_2D.png')
plt.show()

print("✓ Visualizaciones 2D generadas")

# ========== Visualización 3D ==========
print("\n" + "="*80)
print("FASE 13: VISUALIZACIÓN 3D")
print("="*80)

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    cluster_data = df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id]
    ax.scatter(cluster_data['Volumen_Total'],
              cluster_data['Frecuencia_Pedidos'],
              cluster_data['Concentracion_Ventas'],
              c=[colores[cluster_id]],
              label=f"Cluster {cluster_id}: {cluster_names.get(cluster_id, 'Sin def')}",
              s=100, alpha=0.6, edgecolors='black')

ax.set_xlabel('Volumen Total', fontsize=11, fontweight='bold')
ax.set_ylabel('Frecuencia de Pedidos', fontsize=11, fontweight='bold')
ax.set_zlabel('Concentración de Ventas', fontsize=11, fontweight='bold')
ax.set_title('Visualización 3D de Clusters\n(Volumen vs Frecuencia vs Concentración)', 
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r'c:\Users\USER\Downloads\leccion\05_Clusters_3D.png')
plt.show()

print("✓ Visualización 3D generada")

# ========== Conclusiones ==========
print("\n" + "="*80)
print("FASE 14: CONCLUSIONES Y RECOMENDACIONES")
print("="*80)

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                      RESUMEN EJECUTIVO                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

MODELO SELECCIONADO: K-means con K={k_optimal}
  • Silhouette Score: {silhouette_kmeans_final:.4f}
  • Inércia: {inertia_kmeans_final:.2f}
  • Razón: Mayor cohesión y mejor interpretabilidad para el negocio

CLUSTERS IDENTIFICADOS: {k_optimal} segmentos de productos
""")

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    nombre = cluster_names.get(cluster_id, 'Sin definir')
    cantidad = len(df_clusters_pandas[df_clusters_pandas['Cluster_KMeans'] == cluster_id])
    print(f"  • Cluster {cluster_id}: '{nombre}' ({cantidad} productos)")

print(f"""
ESTRATEGIAS DE GESTIÓN DE INVENTARIO:
""")

for cluster_id in sorted(df_clusters_pandas['Cluster_KMeans'].unique()):
    nombre = cluster_names.get(cluster_id, 'Sin definir')
    print(f"\n  ┌─ CLUSTER {cluster_id}: {nombre}")
    
    if "Alta Rotación Centralizada" in nombre:
        print(f"  │  Almacenamiento: Zona de fácil acceso (Área A1)")
        print(f"  │  Replenishment: Semanal/bi-semanal, volúmenes grandes")
        print(f"  │  KPIs: Tasa ruptura < 2%, Obsolescencia < 1%")
    elif "Alta Rotación Dispersa" in nombre:
        print(f"  │  Almacenamiento: Zona de acceso fácil (Área A2)")
        print(f"  │  Replenishment: Semanal, volúmenes medianos")
        print(f"  │  KPIs: Tasa ruptura < 3%, Eficiencia pick/pack > 95%")
    elif "Baja Rotación" in nombre:
        print(f"  │  Almacenamiento: Zona profunda (Área C)")
        print(f"  │  Replenishment: Mensual o bajo demanda")
        print(f"  │  KPIs: Obsolescencia < 10%, Minimizar costos almacenamiento")
    else:
        print(f"  │  Almacenamiento: Zona media (Área B)")
        print(f"  │  Replenishment: Bi-semanal")
        print(f"  │  KPIs: Tasa ruptura < 5%, Cobertura > 90%")
    print(f"  └─")

print(f"""
LIMITACIONES DEL ESTUDIO:
  ⚠ Datos limitados a 2 semanas (semanas 10-11)
  ⚠ No captura estacionalidad o picos de demanda
  ⚠ Variables externas no consideradas (precio, promociones)

RECOMENDACIONES PARA PRODUCCIÓN:
  → Extender análisis a 12+ semanas de datos
  → Incluir variables: costo del producto, margen, promociones
  → Validar con expertos de logística
  → Monitorear cambios y re-clustering mensual
  → Implementar alertas automáticas para transiciones

╔════════════════════════════════════════════════════════════════════════════╗
║                    ✓ ANÁLISIS COMPLETADO EXITOSAMENTE                    ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

# ========== Guardar Resultados ==========
print("\n" + "="*80)
print("GUARDANDO RESULTADOS")
print("="*80)

df_clusters_pandas.to_csv(r'c:\Users\USER\Downloads\leccion\resultados_clustering.csv', index=False)
print(f"\n✓ Archivo guardado: resultados_clustering.csv")
print(f"  • Total de registros: {len(df_clusters_pandas)}")
print(f"  • Columnas: {', '.join(df_clusters_pandas.columns.tolist())}")
print(f"  • Ubicación: c:\\Users\\USER\\Downloads\\leccion\\")

print(f"\n✓ Gráficos guardados:")
print(f"  • 01_Distribuciones.png")
print(f"  • 02_K_Optimo.png")
print(f"  • 03_GMM_BIC.png")
print(f"  • 04_Clusters_2D.png")
print(f"  • 05_Clusters_3D.png")

print("\n" + "="*80)
print("✓ PROCESO FINALIZADO")
print("="*80)

spark.stop()
print("\n✓ Spark Session cerrada")
