set dotenv-load

default: install

install:
	poetry install --no-root

migrate:
	PYTHONPATH=. poetry run python db/run.py up

setup: db-down db-up migrate write clean

db-up:
	docker compose up -d 
	until docker compose exec postgres pg_isready -U sandworm -d queries; do sleep 1; done

db-down:
	docker compose down

clean:
	PYTHONPATH=. poetry run python scripts/clean_data.py

extract limit="1000000":
	PYTHONPATH=. poetry run python scripts/run_extractor.py {{ if limit != "" { "--limit " + limit } else { "" } }}

write:
	PYTHONPATH=. poetry run python scripts/write_to_db.py

popular-tables:
	PYTHONPATH=. poetry run python scripts/popular_tables.py

popular-combos:
	PYTHONPATH=. poetry run python scripts/popular_combos.py

token-count:
	PYTHONPATH=. poetry run python scripts/token_count.py

aidelml-visual:
	PYTHONPATH=. streamlit run aidelml_visual/app.py

extract_tables:
	PYTHONPATH=. poetry run python domains/scripts/extract_source_tables.py --format source_table --unique > processed_sources.csv

extract_tables_domain:
	PYTHONPATH=. poetry run python table_domain/domains.py 