.PHONY: install test smoke status guide clean

install:
	python -m pip install -e . --no-build-isolation

test:
	PYTHONPATH=src pytest -q

smoke:
	bash scripts/run_smoke.sh

status:
	bash scripts/status.sh

guide:
	cd guide && pdflatex -interaction=nonstopmode QTRC_Stop_Huong_dan_IEEE1cot.tex && pdflatex -interaction=nonstopmode QTRC_Stop_Huong_dan_IEEE1cot.tex

clean:
	rm -rf outputs/smoke .pytest_cache src/qtrc_stop.egg-info
