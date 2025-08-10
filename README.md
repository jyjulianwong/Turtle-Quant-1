# Turtle Quant 1

<img src="docs/turtle.webp" alt="drawing" style="width:96px;"/>

## Tech stack

[![My Skills](https://skillicons.dev/icons?i=docker,gcp,githubactions,py,sklearn,terraform)](https://skillicons.dev)

## TL;DR — What's so cool about this repo anyways?

- **Technical indicators**: A widely varied collection of modular hand-selected momentum and candlestick pattern indicators that provide early detection of changes in the direction of price trends
- **Support / resistance indicators**: A varied set of support and resistance calculation algorithms that offer high accuracy and confidence through a fair voting mechanism, helping identify potential trading opportunities for other indicator-based strategies
- **Parallelisation**: Efficient use of Python's multiprocessing and multithreading libraries to run strategy calculations in parallel, increasing throughput
- **Memoization**: Efficient caching of computationally-expensive indicator calculations using thread-safe file-based caches to speed up backtesting and live predictions
- **Cloud orchestration**: Use of Google Cloud Run Jobs that execute long-running tasks in parallel for better throughput and responsiveness
- **DevOps**: A one-click push-to-main CI/CD pipeline trigger that builds and deploys Docker images and Cloud Run Jobs as part of an efficient DevOps setup using GitHub Actions and Terraform
- **Software design**: Loosely-coupled and cohesive classes that: 1. adhere to object-oriented programming and software design best practices, 2. offer simple plug-and-play multi-vendor support (e.g. Yahoo Finance, Alpha Vantage)

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
