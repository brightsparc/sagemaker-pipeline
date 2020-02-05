import sys
import time
import json
import os
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, CollectionConfig

## Arguments ##

bucket_name = sys.argv[1]
prefix = sys.argv[2]
execution_role = sys.argv[3]

## Create baseline

start = time.time()
print('Starting monitor baseline')

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

baseline_data_path = 's3://{0}/{1}/monitoring/baselining/data'.format(bucket_name, prefix)
baseline_results_path = 's3://{0}/{1}/monitoring/baselining/results'.format(bucket_name, prefix)

print(baseline_data_path)
print(baseline_results_path)

my_default_monitor = DefaultModelMonitor(
    role=execution_role,
    instance_count=1,
    instance_type='ml.c5.4xlarge',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_path,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_path,
    wait=True
)

# Export processing job as env variables
# see: https://docs.aws.amazon.com/codepipeline/latest/userguide/reference-variables.html

processing_job_name = my_default_monitor.latest_baselining_job_name
os.environ['PROCESSING_JOB_NAME'] = processing_job_name

end = time.time()
print('Monitor baseline complete in: {}'.format(end - start))