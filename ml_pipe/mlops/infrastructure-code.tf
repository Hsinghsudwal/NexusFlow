# main.tf - Terraform setup for cloud infrastructure

provider "aws" {
  region = "us-west-2"
}

# EKS Cluster for ML training
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "mlops-cluster"
  cluster_version = "1.25"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups for ML workloads
  eks_managed_node_groups = {
    gpu = {
      desired_size = 1
      max_size     = 5
      min_size     = 0
      
      instance_types = ["p3.2xlarge"]
      capacity_type  = "SPOT"
      
      labels = {
        workload = "training"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
    
    cpu = {
      desired_size = 2
      max_size     = 10
      min_size     = 1
      
      instance_types = ["c5.2xlarge"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        workload = "inference"
      }
    }
  }
}

# S3 for model registry and datasets
resource "aws_s3_bucket" "model_registry" {
  bucket = "mlops-model-registry"
  acl    = "private"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    enabled = true
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
  }
}

# RDS for metadata storage
resource "aws_db_instance" "metadata_store" {
  allocated_storage    = 20
  storage_type         = "gp2"
  engine               = "postgres"
  engine_version       = "13.4"
  instance_class       = "db.t3.medium"
  name                 = "mlops_metadata"
  username             = "mlops_admin"
  password             = var.db_password
  parameter_group_name = "default.postgres13"
  skip_final_snapshot  = true
  
  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "mon:04:00-mon:05:00"
}

# ElastiCache for feature store
resource "aws_elasticache_cluster" "feature_store" {
  cluster_id           = "mlops-feature-store"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis6.x"
  engine_version       = "6.x"
  port                 = 6379
}
