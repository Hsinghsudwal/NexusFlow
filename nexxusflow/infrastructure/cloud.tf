provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "artifact_store" {
  bucket = "mlops-artifact-store"
}

resource "aws_dynamodb_table" "metadata" {
  name         = "mlops_metadata"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
}
