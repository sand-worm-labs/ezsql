import aide
import logging

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("aide").setLevel(logging.INFO)

    exp = aide.Experiment(
        data_dir="./grimoire_eval",
        goal=(
            "Classify Dune SQL queries into a g1-g5 taxonomy. "
            "g3 is the canonical structural tool identity: max 4 words, snake_case, "
            "no protocol names, chain names, token names, addresses, or dates. "
            "Use the ground truth labels in train.csv to learn the pattern, "
            "then predict g3 for queries in test.csv."
        ),
        eval="F1 score on g3 label accuracy",
    )

    best_solution = exp.run(steps=30)
    print(f"Best metric: {best_solution.valid_metric}")
    print(f"Best code:\n{best_solution.code}")

if __name__ == "__main__":
    main()