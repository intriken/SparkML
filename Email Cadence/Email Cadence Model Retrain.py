import os
from os.path import expanduser, join, abspath
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    MinMaxScaler,
    QuantileDiscretizer,
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.sql.types import *
from pyspark.sql.functions import lit, udf, col, when, log, sum
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from datetime import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from email.mime.text import MIMEText
from subprocess import Popen, PIPE
from subprocess import call
from datetime import datetime, timedelta
import time
import datetime
from dateutil.relativedelta import relativedelta

os.system("hadoop fs -rm -R -skipTrash /user/user/.Trash")

warehouse_location = abspath("/user/user/user_hiveDB.db")

spark = (
    SparkSession.builder.master("yarn")
    .config("spark.app.name", "email_cadence_model_retrain")
    .config("spark.driver.maxResultSize", "40G")
    .config("spark.driver.memory", "128G")
    .config("spark.dynamicAllocation.enabled", "false")
    .config("spark.executor.cores", 5)
    .config("spark.executor.instances", 50)
    .config("spark.executor.memory", "60G")
    .config("spark.kryoserializer.buffer.max", "1024M")
    .config("spark.network.timeout", "800s")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .config("spark.shuffle.service.enabled", "true")
    .config("spark.sql.hive.convertMetastoreOrc", "false")
    .config(
        "spark.yarn.dist.files",
        "/var/test/spark-2.0.1/python/lib/pyspark.zip,/var/test/spark-2.0.1/python/lib/py4j-0.10.3-src.zip",
    )
    .config(
        "spark.yarn.dist.archives",
        "/var/test/spark-2.0.1/R/lib/sparkr.zip#sparkr,hdfs:////user/grp_gdoop_admin/anaconda/anaconda2_env.zip#ANACONDA",
    )
    .config("spark.executorEnv.PYTHONPATH", "pyspark.zip:py4j-0.10.3-src.zip")
    .config("spark.yarn.executor.memoryOverhead", "8192")
    .config("spark.yarn.queue", "public")
    .config("spark.sql.warehouse.dir", warehouse_location)
    .enableHiveSupport()
    .getOrCreate()
)

spark.conf.set("spark.sql.orc.filterPushdown", "true")

sc = SparkContext.getOrCreate()

sqlContext = HiveContext(sc)

fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())


# paragraph 2 - email send function and dates
def send_mail(message, to, subject):
    msg = MIMEText(message)
    msg["From"] = "user@server3.snc1"
    msg["To"] = to
    msg["Subject"] = subject
    p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE)
    p.communicate(msg.as_string())


today = datetime.date.today()

sdate = datetime.date.today() - relativedelta(days=39)
edate = datetime.date.today() - relativedelta(days=9)

t1 = time.time()
errormsg = ""
testinfo = ""

print("Today: " + str(today))
print("Start Date: " + str(sdate))
print("End Date: " + str(edate))


# paragraph 3 - send start email
send_mail(
    "Email Cadence Model Retrain Starting",
    "pn-core@test.com",
    "Email Cadence Model Retrain: " + str(today),
)


# paragraph 4 - pulling data for 30 days and taking 1%
try:
    sqlContext = HiveContext(sc)
    query = (
        "SELECT a.* FROM (select * from test.features where ds between '"
        + str(sdate)
        + "' and '"
        + str(edate)
        + "') as a where rand() <= 0.01 distribute by rand() sort by rand()"
    )

    print(query)

    data = sqlContext.sql(query)

    data = data.filter(col("sends_30d") > 0)

    cols = data.select(
        "user_uuid",
        "ds",
        "9block_segment",
        "days_since_subscription",
        "opens_30d",
        "clicks_30d",
        "unsubs_180d",
        "purchases_30d",
        "nob_30d",
        "page_views",
        "unique_deal_views",
        "unique_buy_button_clks",
        "unique_cart_chkout_views",
        "total_nob_30d",
        "total_gp_30d",
        "total_purchases_30d",
        "total_pds_purchased",
        "total_ils_purchases_30d",
        "push_nob_30d",
        "30d_visits_app",
        "30d_visits_web",
        "30d_visits_touch",
        "l2_fd_visits_30",
        "l2_ttd_visits_30",
        "l2_retail_visits_30",
        "l2_hbw_visits_30",
        "l2_ha_visits_30",
        "l2_goods_visits",
        "l2_travel_visits_30",
        "predicted_cv",
        "365d_customer_savings",
        "days_since_last_purchase_web",
        "days_since_last_purchase_app",
        "page_views_7d",
        "deal_views_7d",
        "bb_clicks_7d",
        "cart_chkout_7d",
        "search_page_views",
        "next_7d_page_views",
        "next_7d_opens",
        "page_views_1d",
        "deal_views_1d",
        "bb_clicks_1d",
        "cart_chkout_1d",
    ).columns

    data = sqlContext.sql(query)
    data = data.select(cols)
    data = data.fillna("U")
    data = data.fillna(0)
    data = data.filter(col("next_7d_page_views") < 1000)
    data = data.filter(col("next_7d_page_views") >= 0)


except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed retreiving data set for 30 days\n"


# paragraph 5 - configuring data
try:
    replace_infs_udf = udf(
        lambda x, v: float(v) if x and np.isinf(x) else x, DoubleType()
    )

    print("Started adding columns")

    data = data.withColumn(
        "page_view_ctr_7d", data["deal_views_7d"] / data["page_views_7d"]
    ).fillna(0)
    data = data.withColumn(
        "deal_view_ctr_7d", data["bb_clicks_7d"] / data["deal_views_7d"]
    ).fillna(0)
    data = data.withColumn(
        "bbctr_7d", data["cart_chkout_7d"] / data["bb_clicks_7d"]
    ).fillna(0)
    data = data.withColumn(
        "search_dv_rate", data["deal_views_7d"] / data["search_page_views"]
    ).fillna(0)
    data = data.withColumn(
        "next_7d_page_views", f.when(data.next_7d_page_views > 0, 1).otherwise(0)
    )

    print("Finshed adding columns")

    features_to_keep = data.select(
        "days_since_subscription",
        "opens_30d",
        "clicks_30d",
        "unsubs_180d",
        "purchases_30d",
        "nob_30d",
        "page_views",
        "unique_deal_views",
        "unique_buy_button_clks",
        "unique_cart_chkout_views",
        "total_nob_30d",
        "total_gp_30d",
        "total_purchases_30d",
        "total_pds_purchased",
        "total_ils_purchases_30d",
        "push_nob_30d",
        "30d_visits_app",
        "30d_visits_web",
        "30d_visits_touch",
        "l2_fd_visits_30",
        "l2_ttd_visits_30",
        "l2_retail_visits_30",
        "l2_hbw_visits_30",
        "l2_ha_visits_30",
        "l2_goods_visits",
        "l2_travel_visits_30",
        "predicted_cv",
        "365d_customer_savings",
        "days_since_last_purchase_web",
        "days_since_last_purchase_app",
        "page_views_7d",
        "deal_views_7d",
        "bb_clicks_7d",
        "cart_chkout_7d",
        "search_page_views",
        "page_views_1d",
        "deal_views_1d",
        "bb_clicks_1d",
        "cart_chkout_1d",
        "page_view_ctr_7d",
        "deal_view_ctr_7d",
    ).columns

    print("Narrowed down columns")

    assembler = VectorAssembler(inputCols=features_to_keep, outputCol="features")
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed Scaling features\n"

# paragraph 6 - training and evaluate model
try:
    print("Training Model")
    train, test = data.randomSplit([0.7, 0.3], seed=2019)
    # print("Training Dataset Count: " + str(train.count()))
    # print("Test Dataset Count: " + str(test.count()))

    train = train.repartition(500)
    test = test.repartition(500)

    ##trains both random forest and log reg. LR used for scoring because it performed better in GP simulation on new data
    lr_user_val = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="next_7d_page_views",
        maxIter=100,
        regParam=0.001,
        elasticNetParam=0.7,
    )
    pipeline_lr = Pipeline(stages=[assembler] + [scaler] + [lr_user_val])
    lr_model = pipeline_lr.fit(train)

    train = train.withColumn(
        "next_7d_page_views", f.when(train.next_7d_page_views > 0, 1.0).otherwise(0.0)
    )
    test = test.withColumn(
        "next_7d_page_views", f.when(test.next_7d_page_views > 0, 1.0).otherwise(0.0)
    )
    train = train.withColumn(
        "next_7d_page_views", train["next_7d_page_views"].cast(DoubleType())
    )
    test = test.withColumn(
        "next_7d_page_views", test["next_7d_page_views"].cast(DoubleType())
    )

    ##rf = RandomForestClassifier(labelCol='next_7d_page_views',featuresCol='scaled_features', numTrees=100, maxDepth = 9)
    ##pipeline_rf = Pipeline(stages=[assembler]+[scaler]+[rf])
    ##rf_model = pipeline_rf.fit(train)
    ##rf_predictions = rf_model.transform(test)

    print("Evaluating Model")
    vector_udf = udf(lambda vector: float(vector[1]), DoubleType())
    lr_predictions = lr_model.transform(test)

    evaluator = BinaryClassificationEvaluator().setLabelCol("next_7d_page_views")

    ##testinfo = testinfo + "RF Test Area Under PR: " + str(evaluator.evaluate(rf_predictions, {evaluator.metricName: "areaUnderPR"})) + "\n"
    ##testinfo = testinfo + "RF Test Area Under ROC: " + str(evaluator.evaluate(rf_predictions, {evaluator.metricName: "areaUnderROC"}))  + "\n"

    testinfo = (
        testinfo
        + "LR Test Area Under PR: "
        + str(evaluator.evaluate(lr_predictions, {evaluator.metricName: "areaUnderPR"}))
        + "\n"
    )
    testinfo = (
        testinfo
        + "LR Test Area Under ROC: "
        + str(
            evaluator.evaluate(lr_predictions, {evaluator.metricName: "areaUnderROC"})
        )
        + "\n"
    )

    print(testinfo)
    print("Finished")

except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed creating model\n"

# paragraph 7 - backup old model
try:
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path

    todaystringnodash = string = str(today).replace("-", "")

    savepath = "/user/datascience/test_db.db/features/lrModel"

    backuppath = savepath + "_" + todaystringnodash

    print(backuppath)

    backupsuc = fs.rename(Path(savepath), Path(backuppath))

    if backupsuc:
        print("Backuped Model to " + backuppath)
    else:
        print("Failed to Backuped Model to " + backuppath)

    errormsg = ""
except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed backing up model in hdfs\n"

# paragraph 8 - output model to hdfs

try:
    lr_model.write().overwrite().save(savepath)

    print("Saved Model to /user/datascience/test_db.db/features/lrModel")
    errormsg = ""
except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed saving model to hdfs\n"

# paragraph 9 - write coeff to hive
try:
    coef = [i for i in lr_model.stages[2].coefficients]
    feats = [i for i in lr_model.stages[0].getInputCols()]

    coefdata = pd.DataFrame(coef)
    coefdata.columns = ["coef"]
    coefdata["feats"] = feats
    coefdata["abs_coef"] = abs(coefdata.coef)
    coefdata.sort_values("abs_coef", ascending=False)

    df_coef = spark.createDataFrame(coefdata)
    df_coef = df_coef.withColumn("retrain_date", lit(today))

    df_coef = df_coef.select("feats", "coef", "abs_coef", "retrain_date")

    df_coef.createOrReplaceTempView("coef_temp")

    sqlContext.sql("set hive.exec.dynamic.partition.mode=nonstrict")
    sqlContext.sql(
        "insert overwrite table test.email_cadence_page_view_model_scores_coefficients partition(retrain_date) select * from coef_temp"
    )

except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed saving coefficients to table\n"


# paragraph 10 - send final email
temp = time.time() - t1

hours = temp // 3600
temp = temp - 3600 * hours
minutes = temp // 60
seconds = temp - 60 * minutes
time_str = "hours:" + str(hours) + "; mins " + str(minutes) + "; secs " + str(seconds)
print(errormsg)

if errormsg == "":
    send_mail(
        "Email Cadence Model Retrain Completed in " + time_str + "\n\n" + testinfo,
        "pn-core@test.com",
        "Email Cadence Model Retrain: " + str(today),
    )
else:
    send_mail(
        "Email Cadence Model Retrain Failed in " + time_str + "\n\n" + errormsg,
        "pn-core@test.com",
        "Email Cadence Model Retrain: " + str(today) + ": FAILED",
    )


# paragraph 11 - stop spark
sc.stop()
