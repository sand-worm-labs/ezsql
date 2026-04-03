set dotenv-load

default: install

install:
	poetry install --no-root

migrate:
	PYTHONPATH=. poetry run python db/run.py up

setup: db-down db-up migrate write clean 

db-up:
	docker compose up -d postgres
	until docker compose exec postgres pg_isready -U sandworm -d queries; do sleep 1; done

db-down:
	docker compose down

clean:
	PYTHONPATH=. poetry run python scripts/clean_data.py

write:
	PYTHONPATH=. poetry run python scripts/write_to_db.py
