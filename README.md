# Fraud Detection Classification

## Machine Learning Application on GCP

## Deploy

* ssh into GCP Virtual Machine
```
ssh user_name@external_ip
git clone {this repo}
```

* install pip/pip3 on gcp
```
pip install -r requirements.txt
```

* virtual env setup
```
virtualenv --python=python3 venv
source venv/bin/activate
```

* run flask
```
python main.py
```

* exit virtual env
```
deactivate
```
