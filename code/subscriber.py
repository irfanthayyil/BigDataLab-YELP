from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def eval_metrics(df, epoch_id):

    if df.count() > 0:
        eval_acc =  MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        eval_f1 =  MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        print("-"*50)
        print(f"Batch: {epoch_id}")
        print("-"*50)
        df.show(df.count())
        print(f"Accuracy: {eval_acc.evaluate(df):.4f}\nF1 score: {eval_f1.evaluate(df):.4f}")

    pass


def get_inferences(broker_ip, topic, model_path):
    
    spark = SparkSession.builder.appName("yelp_proj").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.readStream.format("kafka").\
            option("kafka.bootstrap.servers", f"{broker_ip}:9092").\
            option("subscribe", topic).\
            load()
    
    split_cols = F.split(df.value,'\t')
    df = df.withColumn('cool',split_cols.getItem(1))
    df = df.withColumn('funny',split_cols.getItem(3))
    df = df.withColumn('stars',split_cols.getItem(5))
    df = df.withColumn('text',split_cols.getItem(6))
    df = df.withColumn('useful',split_cols.getItem(7))
    
    for col in ['cool', 'funny', 'stars', 'useful']:
        df = df.withColumn(col, df[col].cast('float'))

    df = df.withColumnRenamed("stars","label")
    
    df.createOrReplaceTempView("intermediate")
    
    model = PipelineModel.load(model_path)
    
    predictions = model.transform(df)
    
    predictions = predictions.withColumn('correct',F.when((F.col('prediction')== F.col('label')),1).otherwise(0))

    output_df = predictions[['prediction', 'label', 'correct']]
    output_df.createOrReplaceTempView('output')
    
    query = output_df.writeStream.foreachBatch(eval_metrics).start()
    query.awaitTermination()

if __name__=="__main__":
    
    BROKER_IP = "10.128.0.8"
    TOPIC = "nlp"
    DATA_DIR = "gs://sgb1/yelp_train.json"
    MODEL_DIR = "gs://sgb1/PipelineModel_LR"

    get_inferences(BROKER_IP, TOPIC, MODEL_DIR)
