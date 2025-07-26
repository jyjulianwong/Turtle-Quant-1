# Turtle Quant 1

<img src="docs/turtle.webp" alt="drawing" style="width:96px;"/>

## Tech stack

[![My Skills](https://skillicons.dev/icons?i=docker,gcp,githubactions,py,terraform)](https://skillicons.dev)

## Get started with development

1. Clone the repository.

```bash
git clone https://github.com/jyjulianwong/Turtle-Quant-1.git
```

2. Verify that you have a compatible Python version installed on your machine.
```bash
python --version
```

3. Install [uv](https://github.com/astral-sh/uv) (used as the package manager for this project).

4. Install the development dependencies.
```bash
cd Turtle-Quant-1/
uv sync --all-groups
uv run pre-commit install
```

(TBC.)

## Deployment

For naming conventions, refer to https://stepan.wtf/cloud-naming-convention/.

### Continuous deployment

(TBC.)

## Google Cloud administration

### Setting IAM permissions and roles

```bash
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value core/project)
gcloud iam service-accounts create svc-usea1-tf
gcloud iam service-accounts keys create ~/key.json --iam-account svc-usea1-tf@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} --member "serviceAccount:svc-usea1-tf@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" --role "roles/bigquery.user"
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
```
