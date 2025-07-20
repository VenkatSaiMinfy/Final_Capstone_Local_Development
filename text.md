CREATE TABLE public.lead_data (
    "Prospect ID" text,
    "Lead Number" bigint,
    "Lead Origin" text,
    "Lead Source" text,
    "Do Not Email" text,
    "Do Not Call" text,
    "Converted" bigint,
    "TotalVisits" double precision,
    "Total Time Spent on Website" bigint,
    "Page Views Per Visit" double precision,
    "Last Activity" text,
    "Country" text,
    "Specialization" text,
    "How did you hear about X Education" text,
    "What is your current occupation" text,
    "What matters most to you in choosing a course" text,
    "Search" text,
    "Magazine" text,
    "Newspaper Article" text,
    "X Education Forums" text,
    "Newspaper" text,
    "Digital Advertisement" text,
    "Through Recommendations" text,
    "Receive More Updates About Our Courses" text,
    "Tags" text,
    "Lead Quality" text,
    "Update me on Supply Chain Content" text,
    "Get updates on DM Content" text,
    "Lead Profile" text,
    "City" text,
    "Asymmetrique Activity Index" text,
    "Asymmetrique Profile Index" text,
    "Asymmetrique Activity Score" double precision,
    "Asymmetrique Profile Score" double precision,
    "I agree to pay the amount through cheque" text,
    "A free copy of Mastering The Interview" text,
    "Last Notable Activity" text
);



import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue import DynamicFrame

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Script generated for node Amazon S3
AmazonS3_node1752777278258 = glueContext.create_dynamic_frame.from_catalog(database="capstone database", table_name="lead_scoring_csv", transformation_ctx="AmazonS3_node1752777278258")

# Script generated for node Amazon Redshift
AmazonRedshift_node1752777280887 = glueContext.write_dynamic_frame.from_options(
    frame=AmazonS3_node1752777278258,
    connection_type="redshift",
    connection_options={
        "redshiftTmpDir": "s3://aws-glue-assets-122610482767-ap-south-1/temporary/",
        "useConnectionProperties": "true",
        "dbtable": "public.lead_data",
        "connectionName": "Redshift connection",
        "preactions": """CREATE TABLE IF NOT EXISTS public.lead_data (
            "Prospect ID" VARCHAR,
            "Lead Number" BIGINT,
            "Lead Origin" VARCHAR,
            "Lead Source" VARCHAR,
            "Do Not Email" VARCHAR,
            "Do Not Call" VARCHAR,
            "Converted" BIGINT,
            "TotalVisits" BIGINT,
            "Total Time Spent on Website" BIGINT,
            "Page Views Per Visit" DOUBLE PRECISION,
            "Last Activity" VARCHAR,
            "Country" VARCHAR,
            "Specialization" VARCHAR,
            "How did you hear about X Education" VARCHAR,
            "What is your current occupation" VARCHAR,
            "What matters most to you in choosing a course" VARCHAR,
            "Search" VARCHAR,
            "Magazine" VARCHAR,
            "Newspaper Article" VARCHAR,
            "X Education Forums" VARCHAR,
            "Newspaper" VARCHAR,
            "Digital Advertisement" VARCHAR,
            "Through Recommendations" VARCHAR,
            "Receive More Updates About Our Courses" VARCHAR,
            "Tags" VARCHAR,
            "Lead Quality" VARCHAR,
            "Update me on Supply Chain Content" VARCHAR,
            "Get updates on DM Content" VARCHAR,
            "Lead Profile" VARCHAR,
            "City" VARCHAR,
            "Asymmetrique Activity Index" VARCHAR,
            "Asymmetrique Profile Index" VARCHAR,
            "Asymmetrique Activity Score" BIGINT,
            "Asymmetrique Profile Score" BIGINT,
            "I agree to pay the amount through cheque" VARCHAR,
            "A free copy of Mastering The Interview" VARCHAR,
            "Last Notable Activity" VARCHAR
        );"""
    },
    transformation_ctx="AmazonRedshift_node1752777280887"
)

job.commit()










# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/

import boto3
from botocore.exceptions import ClientError


def get_secret():

    secret_name = "capstonesecret"
    region_name = "ap-south-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = get_secret_value_response['SecretString']

    # Your code goes here.
