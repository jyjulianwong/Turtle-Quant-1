# Google Cloud Run Service
resource "google_cloud_run_service" "app" {
  name     = "${var.turtlequant1_gcloud_project_id}-run-${var.turtlequant1_gcloud_region}-app"
  location = var.turtlequant1_gcloud_region

  template {
    metadata {
      annotations = {
        "autoscaling.knative.dev/maxScale" = "5"
        "custom/revision-suffix"           = substr(md5(timestamp()), 0, 4)
      }
    }

    spec {
      containers {
        image = var.image_tag

        resources {
          limits = {
            memory = "2Gi"
            cpu    = "1000m"
          }
        }

        env {
          name  = "TURTLEQUANT1_ENV"
          value = var.turtlequant1_env
        }

        env {
          name  = "TURTLEQUANT1_GCLOUD_REGION"
          value = var.turtlequant1_gcloud_region
        }

        env {
          name  = "TURTLEQUANT1_GCLOUD_PROJECT_ID"
          value = var.turtlequant1_gcloud_project_id
        }

        env {
          name  = "TURTLEQUANT1_GCLOUD_STB_DATA_NAME"
          value = var.turtlequant1_gcloud_stb_data_name
        }

        env {
          name  = "TURTLEQUANT1_ALPHA_VANTAGE_API_KEY"
          value = var.turtlequant1_alpha_vantage_api_key
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Allow unauthenticated access to Cloud Run service
resource "google_cloud_run_service_iam_member" "app_public" {
  service  = google_cloud_run_service.app.name
  location = google_cloud_run_service.app.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Google Cloud Storage for data storage
resource "google_storage_bucket" "data" {
  name     = var.turtlequant1_gcloud_stb_data_name
  location = var.turtlequant1_gcloud_region

  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90 # days
    }
    action {
      type = "Delete"
    }
  }
}
