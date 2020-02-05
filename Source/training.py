import sys
import time
import json
import boto3
import os
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker.debugger import Rule, rule_configs, DebuggerHookConfig, CollectionConfig

## Arguments ##

bucket_name = sys.argv[1]
prefix = sys.argv[2]
execution_role = sys.argv[3]
autoscaling_role = sys.argv[4]
exp_name = sys.argv[5]
trial_name = sys.argv[6][0:7]

## Create Experiment and Trial

start = time.time()
print('Creating experiment: {} and trial: {}'.format(exp_name, trial_name))

sm = boto3.client('sagemaker')

try:
    response = sm.create_experiment(
        ExperimentName=exp_name,
        DisplayName=exp_name,
        Description='MLOps experiment'
    )
    print("Created experiment: %s" % response)
except ClientError as e:
    if e.response['Error']['Code'] == 'ValidationException':
        print("Experiment %s already exists" % exp_name)
    else:
        print("Unexpected error: %s" % e)

try:
    response = sm.create_trial(
        TrialName=trial_name,
        DisplayName=trial_name,
        ExperimentName=exp_name,
    )
    print("Created trial: %s" % response)
except ClientError as e:
    if e.response['Error']['Code'] == 'ValidationException':
        print("Trial %s already exists" % trial_name)
    else:
        print("Unexpected error: %s" % e)

## Training ##

hyperparameters = {
    "max_depth": "10",
    "eta": "0.2",
    "gamma": "1",
    "min_child_weight": "6",
    "silent": "0",
    "objective": "multi:softmax",
    "num_class": "15",
    "num_round": "20"
}

entry_point='train_xgboost.py'
source_dir='Source/'
output_path = 's3://{0}/{1}/output'.format(bucket_name, prefix)
debugger_output_path = 's3://{0}/{1}/output/debug'.format(bucket_name, prefix) # Path where we save debug outputs
code_location = 's3://{0}/{1}/code'.format(bucket_name, prefix)

hook_config = DebuggerHookConfig(
    s3_output_path=debugger_output_path,
    hook_parameters={
        "save_interval": "1"
    },
    collection_configs=[
        CollectionConfig("hyperparameters"),
        CollectionConfig("metrics"),
        CollectionConfig("predictions"),
        CollectionConfig("labels"),
        CollectionConfig("feature_importance")
    ]
)

job_name_base = "{}-{}".format(exp_name, trial_name)

estimator = XGBoost(
    base_job_name=job_name_base,
    entry_point=entry_point,
    source_dir=source_dir,
    output_path=output_path,
    code_location=code_location,
    hyperparameters=hyperparameters,
    train_instance_type="ml.m5.4xlarge",
    train_instance_count=1,
    framework_version="0.90-2",
    py_version="py3",
    role=execution_role,
    
    # Initialize your hook.
    debugger_hook_config=hook_config,
    
    # Initialize your rules. These will read data for analyses from the path specified
    # for the hook
    rules=[Rule.sagemaker(rule_configs.confusion(),
                             rule_parameters={
                                 "category_no": "15",
                                 "min_diag": "0.7",
                                 "max_off_diag": "0.3",
                                 "start_step": "17",
                                 "end_step": "19"}
                         )]
)

train_config = sagemaker.session.s3_input('s3://{0}/{1}/data/train/'.format(
    bucket_name, prefix), content_type='text/csv')
val_config = sagemaker.session.s3_input('s3://{0}/{1}/data/val/'.format(
    bucket_name, prefix), content_type='text/csv')

training_image = estimator.image_name

print('Training image: {}'.format(training_image))

# TODO: In future look at create non-blocking SF workflow, and use CW event callbak
# see: https://github.com/brightsparc/amazon-sagemaker-mlops-workshop/blob/master/assets/train-model-pipeline.yml#L115

estimator.fit(
    inputs={'train': train_config, 'validation': val_config },
    experiment_config={
        "ExperimentName": exp_name,
        "TrialName": trial_name,
        "TrialComponentDisplayName": "Training",
    },
    wait=True)

job_name = estimator.latest_training_job.name

end = time.time()
print('Training job {} complete in {}'.format(job_name, end - start))

# save environment variables

os.environ['QA_ENDPOINT_NAME'] = "qa-{}".format(exp_name)
os.environ['PROD_ENDPOINT_NAME'] = "prod-{}".format(exp_name)

# creating configuration files so we can pass parameters to our sagemaker endpoint cloudformation

config_data_qa = {
  "Parameters":
    {
        "ModelName": "qa-{}".format(job_name),
        "EndpointName": "qa-{}".format(exp_name),
        "EndpointConfigName": "qa-{}".format(job_name),
        "ModelDataUrl": "{}/{}/output/model.tar.gz".format(output_path, job_name),
        "SageMakerRole": execution_role,
        "SageMakerImage": training_image
    }
}

config_data_prod = {
  "Parameters":
    {
        "ModelName": "prod-{}".format(job_name),
        "EndpointName": "prod-{}".format(exp_name),
        "EndpointConfigName": "prod-{}".format(job_name),
        "ModelDataUrl": "{}/{}/output/model.tar.gz".format(output_path, job_name),
        "SageMakerRole": execution_role,
        "SageMakerImage": training_image,
        "AutoScalingRole": autoscaling_role 
    }
}

json_config_data_qa = json.dumps(config_data_qa)
json_config_data_prod = json.dumps(config_data_prod)

f = open( './CloudFormation/configuration_qa.json', 'w' )
f.write(json_config_data_qa)
f.close()

f = open( './CloudFormation/configuration_prod.json', 'w' )
f.write(json_config_data_prod)
f.close()