# EzSQL: SQL-to-Text Generation for Blockchain Analytics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EzSQL** is an intermediate representation (IR) system that bridges the gap between SQL queries and natural language. Originally based on the [EzSQL paper](https://arxiv.org/abs/2305.xxxxx), this implementation is specifically adapted for **blockchain analytics queries** from platforms like Dune Analytics.

## 🎯 Key Features

- **SQL Simplification**: Transform complex SQL into a natural-language-aligned intermediate representation
- **SQL-to-Text Generation**: Generate human-readable descriptions of SQL queries
- **Query Clustering**: Group similar queries by intent using EzSQL embeddings
- **Data Augmentation**: Generate synthetic SQL-text pairs for training

## 📊 Why EzSQL?

Raw SQL is structurally far from natural language:

```sql
-- Original Dune SQL (hard to understand)
SELECT T1.token_symbol, SUM(T1.amount_usd) 
FROM dex.trades AS T1 
JOIN tokens.erc20 AS T2 ON T1.token_address = T2.contract_address
WHERE T1.block_time >= DATE_TRUNC('day', NOW() - INTERVAL '7' DAY)
GROUP BY T1.token_symbol
ORDER BY SUM(T1.amount_usd) DESC
LIMIT 10
```

```sql
-- EzSQL IR (closer to natural language)
SELECT token_symbol, total amount_usd
FROM dex trades, tokens erc20
WHERE block_time >= last 7 days
GROUP BY token_symbol
ORDER BY total amount_usd descending
LIMIT 10
```

```
-- Generated Natural Language
"Show the top 10 tokens by total trading volume in USD over the last 7 days"
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EzSQL Pipeline (Revised)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌───────────┐    ┌───────────┐    ┌───────────────────┐  │
│   │ Raw SQL  │───▶│  EzSQL    │───▶│   Text    │───▶│      Intent       │  │
│   │ (Dune)   │    │    IR     │    │Description│    │   Classification  │  │
│   └──────────┘    └───────────┘    └───────────┘    └───────────────────┘  │
│                                                               │              │
│                                                               ▼              │
│                                                    ┌───────────────────┐    │
│                                                    │   Applications    │    │
│                                                    │  • Clustering     │    │
│                                                    │  • Search         │    │
│                                                    │  • Recommendations│    │
│                                                    │  • Data Augment   │    │
│                                                    └───────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sandworm-labs/ezsql.git
cd ezsql

# Install with poetry
poetry install

# Or with pip
pip install -e .
```

### Basic Usage

```python
from ezsql import EzSQLSimplifier, SQLToTextGenerator

# Initialize
simplifier = EzSQLSimplifier()
generator = SQLToTextGenerator()

# Transform SQL to EzSQL IR
sql = """
SELECT token_address, COUNT(*) as transfers
FROM erc20_ethereum.evt_Transfer
WHERE block_time > NOW() - INTERVAL '24 hours'
GROUP BY token_address
ORDER BY transfers DESC
LIMIT 10
"""

ezsql = simplifier.transform(sql)
print(ezsql)
# Output: SELECT token_address, count transfers FROM erc20 ethereum transfers 
#         WHERE block_time > last 24 hours GROUP BY token_address 
#         ORDER BY transfers descending LIMIT 10

# Generate natural language description
description = generator.generate(ezsql)
print(description)
# Output: "Find the top 10 ERC20 tokens by number of transfers in the last 24 hours"
```


## 🛠️ Development

```bash
# Run tests
pytest tests/

# Run linting
ruff check src/

# Format code
ruff format src/

# Build documentation
mkdocs build
```


## 🙏 Acknowledgments

- Original EzSQL paper authors
- Dune Analytics for the query dataset

## 📬 Contact

- **Project**: [Sandworm Labs](https://sandwormlabs.xyz)
- **Issues**: [GitHub Issues](https://github.com/sandworm-labs/ezsql/issues)