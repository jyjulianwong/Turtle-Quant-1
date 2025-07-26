terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.turtlequant1_gcloud_project_id
  region  = var.turtlequant1_gcloud_region
}
