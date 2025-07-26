variable "image_tag" {
  description = "The Docker image tag to deploy to Cloud Run"
  type        = string
}

variable "turtlequant1_env" {
  description = "The environment to deploy to"
  type        = string
}

variable "turtlequant1_gcloud_region" {
  description = "The Google Cloud region to deploy resources to"
  type        = string
  default     = "us-east1"
}

variable "turtlequant1_gcloud_project_id" {
  description = "The Google Cloud project ID"
  type        = string
}

variable "turtlequant1_gcloud_stb_data_name" {
  description = "The name of the Google Cloud Storage bucket for data storage"
  type        = string
}

variable "turtlequant1_alpha_vantage_api_key" {
  description = "The Alpha Vantage API key for market data access"
  type        = string
  sensitive   = true
}
