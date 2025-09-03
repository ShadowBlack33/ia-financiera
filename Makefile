
    .PHONY: venv install etl tests metrics tune notebook

    venv:
	python -m venv .venv

    install: venv
	. .venv/bin/activate && pip install -r requirements.txt

    etl:
	. .venv/bin/activate && python main.py

    tests:
	. .venv/bin/activate && pytest -q

    metrics:
	. .venv/bin/activate && python -c "from models.train_all import run_for_folder; run_for_folder('data/raw')"

    tune:
	. .venv/bin/activate && python -c "from models.tune import tune_file; import glob; [tune_file(p, 'svr', 30) for p in glob.glob('data/raw/*_1d.csv')]"

    notebook:
	@echo "Abre notebooks/metrics_report.ipynb en tu entorno preferido."



    docker-build:
	docker build -t ia-financiera:latest .

    docker-test:
	docker run --rm ia-financiera:latest
