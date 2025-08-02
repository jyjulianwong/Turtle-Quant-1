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

## Get started with Jupyter notebooks

1. Once the above setup is complete, set up a Python kernel.
```bash
source .venv/bin/activate
python -m ipykernel install --user --name=turtle-quant-1
```

2. Refer to the following common commands.
```bash
jupyter kernelspec list
jupyter kernelspec uninstall turtle-quant-1
```

3. Start the Jupyter server.
```bash
jupyter lab
```

## Common entry points

### Backtesting

Run the following script to run all custom backtesting test cases.
```bash
uv run turtle_quant_1/backtesting/main.py
```

### Hyperparameter tuning

Run the following script to run hyperparameter tuning across all strategies.
```bash
uv run turtle_quant_1/backtesting/hyperparameters.py
```

## Deployment

For naming conventions, refer to https://stepan.wtf/cloud-naming-convention/.

### Continuous deployment

Deployment is fully automated and handled by GitHub Actions. Refer to [`.github/workflows/test-build-deploy.yaml`](.github/workflows/test-build-deploy.yaml).

## Google Cloud administration

### Setting IAM permissions and roles

```bash
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value core/project)
gcloud iam service-accounts create svc-usea1-tf
gcloud iam service-accounts keys create ~/key.json --iam-account svc-usea1-tf@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} --member "serviceAccount:svc-usea1-tf@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" --role "roles/bigquery.user"
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
```
