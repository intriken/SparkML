# quartz cron info
# 0 0 2 ? * * *
# every day at 2am utc

# paragraph 1 - imports

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
from pyspark.sql.functions import *
from pyspark.sql.window import Window
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
import getpass

user = getpass.getuser()

os.system("hadoop fs -rm -R -skipTrash /user/{0}/.Trash".format(user))

warehouse_location = abspath("/user/{0}/{0}_hiveDB.db".format(user))

spark = (
    SparkSession.builder.master("yarn")
    .config("spark.app.name", "email_cadence_model")
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
    .config("spark.yarn.queue", "marketing_datascience")
    .config("spark.sql.warehouse.dir", warehouse_location)
    .enableHiveSupport()
    .getOrCreate()
)

spark.conf.set("spark.sql.orc.filterPushdown", "true")

sc = SparkContext.getOrCreate()

sqlContext = HiveContext(sc)

# paragraph 2 email function and dates


def send_mail(message, to, subject):
    msg = MIMEText(message)
    msg["From"] = "svc_push_ds@cerebro-seo-job-submitter2.snc1"
    msg["To"] = to
    msg["Subject"] = subject
    p = Popen(["/usr/sbin/sendmail", "-t", "-oi"], stdin=PIPE)
    p.communicate(msg.as_string())


today = datetime.date.today()
edate = datetime.date.today() - relativedelta(days=2)
t1 = time.time()
errormsg = ""

print(str(edate))


# paragraph 3 - send start email

send_mail(
    "Email Cadence Model Scoring Starting",
    "pn-core@test.com",
    "Email Cadence Model Scoring: " + str(today),
)


# paragraph 4 - loading model from file

try:
    # load model from exported file
    lrModel = PipelineModel.load("/user/datascience/test_db.db/features/lrModel")

    print("Model Loaded")
except Exception, e:
    print(str(e))
    errormsg = errormsg + "Model Failed Loading\n"


# paragraph 5 - data config

###SCORING
print("Scoring")

try:
    ##change date as parameter to take in current date (or date of score)
    scoring_query = "select * from test.features where ds = '" + str(edate) + "'"

    print(scoring_query)
    scoring_data = sqlContext.sql(scoring_query)

    ##scoring_data = scoring_data.filter(col("sends_sd") >3)
    scoring_data = scoring_data.filter(col("sends_30d") > 0)
    scoring_data = scoring_data.fillna("U")
    scoring_data = scoring_data.fillna(0)

    scoring_cols = scoring_data.select(
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

    scoring_data = scoring_data.select(scoring_cols)

    scoring_data = scoring_data.withColumn(
        "page_view_ctr_7d",
        scoring_data["deal_views_7d"] / scoring_data["page_views_7d"],
    ).fillna(0)
    scoring_data = scoring_data.withColumn(
        "deal_view_ctr_7d", scoring_data["bb_clicks_7d"] / scoring_data["deal_views_7d"]
    ).fillna(0)
    scoring_data = scoring_data.withColumn(
        "bbctr_7d", scoring_data["cart_chkout_7d"] / scoring_data["bb_clicks_7d"]
    ).fillna(0)
    scoring_data = scoring_data.withColumn(
        "search_dv_rate",
        scoring_data["deal_views_7d"] / scoring_data["search_page_views"],
    ).fillna(0)

    df_active_emails = spark.sql(
        """
    select consumer_id
    from (select consumer_id, true_user_key
            from (select uuid consumer_id, true_user_key from prod_testdw.dim_user where status = 'active' and current_ind = 1 and user_brand_affiliation ='test' and uuid is not null) m
            join (select user_key from prod_testdw.dim_subscriber_history where status = 'active' and valid_date_end_key = 99991231) a  on a.user_key = m.true_user_key
    ) a
    left join (select user_key from prod_testdw.dim_rested_sub_history where rested_ind = 1 and valid_date_end = '9999-12-31') b on b.user_key = a.true_user_key
    where b.user_key is null
    """
    )

    scoring_data = scoring_data.join(
        df_active_emails,
        scoring_data.user_uuid == df_active_emails.consumer_id,
        "inner",
    ).drop("consumer_id")
    scoring_data = scoring_data.cache()

except Exception, e:
    print(str(e))
    errormsg = errormsg + "Data Failed Loading\n"


# paragraph 6 -  Scoring


try:
    vector_udf = udf(lambda vector: float(vector[1]), DoubleType())

    # create full set of scored data
    scores = lrModel.transform(scoring_data.repartition(400))

    scores = scores.select(
        "user_uuid",
        "ds",
        "9block_segment",
        "total_purchases_30d",
        "total_gp_30d",
        "next_7d_page_views",
        "prediction",
        "probability",
    )
    scores = scores.withColumn("positive_probability", vector_udf(scores.probability))
    scores = scores.cache()

except Exception, e:
    print(str(e))
    errormsg = errormsg + "Probablity Calculation Failed\n"

# paragraph 7 -  Quantile

# try:
#     #create deciles splits
# desired_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # must be sorted
#
#
# quantile_values = scores.approxQuantile("positive_probability",desired_quantiles,0.0001)
#
# print(quantile_values)
#
# except Exception, e:
#     print(str(e))
#     errormsg = errormsg + "Failed Finding Quantile Splits\n"
#
#
# #paragraph 8 -  Quantile Function
#
#
# #defining udf to score deciles based on splits found from sample
# def spec_decile(prob):
#     star = 0
#     decile = 0
#     for i in range(len(quantile_values)):
#         if prob >= star and prob <= quantile_values[i]:
#             return i
#         else:
#             star = quantile_values[i]
#
#     return 9
#
# spec_decile_udf = udf(spec_decile, IntegerType())

# paragraph 9 -  Quantile Addition


try:
    # create deciles for full set
    print("Creating Deciles")

    # prob_scores = scores.withColumn("prob_decile",spec_decile_udf(scores.positive_probability))

    prob_scores = scores.withColumn(
        "prob_decile_temp",
        floor(10 * percent_rank().over(Window.orderBy(scores.positive_probability))),
    )
    prob_scores = prob_scores.withColumn(
        "prob_decile",
        when(col("prob_decile_temp") == 10, 9).otherwise(col("prob_decile_temp")),
    )

except Exception, e:
    print(str(e))
    errormsg = errormsg + "Quantile Failed to Calculate\n"


# paragraph 10 - writing table to hive


try:
    print("Writing to Hive")
    prob_scores = prob_scores.select(
        "user_uuid",
        "9block_segment",
        "total_purchases_30d",
        "total_gp_30d",
        "next_7d_page_views",
        "prediction",
        "positive_probability",
        "prob_decile",
        "ds",
    )

    prob_scores.repartition(50).createOrReplaceTempView("prob_scores")

    print("Adding Partition")
    write_sql1 = "insert overwrite table test.email_cadence_page_view_model_scores partition(ds) select * from prob_scores"
    spark.sql("""set hive.exec.dynamic.partition.mode=nonstrict""")
    spark.sql("""set hive.merge.tezfiles=true""")
    spark.sql("""set hive.merge.smallfiles.avgsize=1024000000""")
    spark.sql(write_sql1)
except Exception, e:
    print(str(e))
    errormsg = errormsg + "Failed Writing table to Hive\n"


# paragraph 11 - send final email

temp = time.time() - t1

hours = temp // 3600
temp = temp - 3600 * hours
minutes = temp // 60
seconds = temp - 60 * minutes
time_str = "hours:" + str(hours) + "; mins " + str(minutes) + "; secs " + str(seconds)
print(errormsg)

if errormsg == "":
    send_mail(
        "Email Cadence Model Scoring Completed in " + time_str,
        "pn-core@test.com",
        "Email Cadence Model Scoring: " + str(today),
    )
else:
    send_mail(
        "Email Cadence Model Scoring Failed in " + time_str + "\n\n" + errormsg,
        "pn-core@test.com",
        "Email Cadence Model Scoring: " + str(today) + ": FAILED",
    )

# paragraph 12 - stop spark

print("Finished")
sc.stop()
