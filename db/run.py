import sys
import importlib

MIGRATIONS = [
    "migrations.001_create_tables",
    "migrations.002_add_pgvector",
    "migrations.003_seed_domains",
    "migrations.004_create_queries_table",
]

def run_up():
    for module_path in MIGRATIONS:
        mod = importlib.import_module(module_path)
        mod.up()
    print("all migrations applied")

def run_down():
    for module_path in reversed(MIGRATIONS):
        mod = importlib.import_module(module_path)
        mod.down()
    print("all migrations rolled back")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"

    if cmd == "up":
        run_up()
    elif cmd == "down":
        run_down()
    elif cmd.startswith("migrations."):
        mod = importlib.import_module(cmd)
        action = sys.argv[2] if len(sys.argv) > 2 else "up"
        getattr(mod, action)()
    else:
        print("usage: python -m migrations.run [up|down]")
        print("       python -m migrations.run migrations.001_create_tables [up|down]")
        sys.exit(1)