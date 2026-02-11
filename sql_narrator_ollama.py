"""
SQL-to-Intent Narrator using Ollama
====================================
Uses local Ollama API for natural language generation.
Falls back to rule-based if Ollama unavailable.
"""

import json
import httpx
from dataclasses import dataclass
from typing import Optional
import sqlglot
from sqlglot import exp
from sqlglot.errors import ErrorLevel

# ========================
# Configuration
# ========================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"  # Small, fast, good at instruction following
# Alternatives: "phi3:mini", "llama3.2:1b", "gemma2:2b", "mistral:7b-instruct-q4_0"

OLLAMA_TIMEOUT = 60.0  # seconds per request
MAX_SQL_LENGTH = 20000  # Truncate very long queries


@dataclass
class OllamaConfig:
    base_url: str = OLLAMA_BASE_URL
    model: str = OLLAMA_MODEL
    timeout: float = OLLAMA_TIMEOUT


# ========================
# Ollama Client
# ========================
class OllamaClient:
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        if self._available is not None:
            return self._available
        
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.config.base_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        
        return self._available
    
    def generate(self, prompt: str, system: str = None) -> str:
        """Generate text using Ollama API."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low for deterministic output
                "num_predict": 2056,  # Max tokens
            }
        }
        
        if system:
            payload["system"] = system
        
        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(
                f"{self.config.base_url}/api/generate",
                json=payload
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()


# ========================
# SQL Component Extractor
# ========================
def extract_sql_components(sql_query: str) -> list[tuple[str, str]]:
    """
    Extract translatable components from SQL.
    Returns list of (label, sql_fragment) tuples.
    """
    try:
        parsed = sqlglot.parse_one(sql_query, error_level=ErrorLevel.IGNORE)
        components = []
        
        # Extract CTEs
        for cte in parsed.find_all(exp.CTE):
            alias = cte.alias
            cte_sql = cte.this.sql(pretty=True)
            components.append((f"Step '{alias}'", cte_sql))
        
        # Extract main query (without CTEs)
        main_query = parsed.copy()
        main_query.set("with", None)
        main_sql = main_query.sql(pretty=True)
        
        if components:
            components.append(("Final result", main_sql))
        else:
            components.append(("Query", main_sql))
        
        return components
    except Exception:
        # Fallback: treat entire query as one component
        return [("Query", sql_query[:MAX_SQL_LENGTH])]


# ========================
# Narrator
# ========================
SYSTEM_PROMPT = """You are a SQL explainer. Given a SQL query or fragment, identify the user's intent and explain it in plain English.
- Be extremely concise (1-2 sentences max)
- Focus on the business purpose, not SQL syntax
- Use simple, clear words
- Do not repeat the SQL back
- Translate technical queries into human-understandable intent
- Think of it as creating a reusable template for the query’s goal
- Capture what the query is trying to achieve, so future queries with a similar structure can be recognized"""

USER_PROMPT_TEMPLATE = """Explain this SQL in plain English:

```sql
{sql}
```

Explanation:"""


class SQLNarrator:
    def __init__(self, ollama_config: OllamaConfig = None):
        self.client = OllamaClient(ollama_config)
        self._ollama_checked = False
        self._use_ollama = False
    
    def _check_ollama(self):
        if not self._ollama_checked:
            self._use_ollama = self.client.is_available()
            self._ollama_checked = True
            if self._use_ollama:
                print(f"✓ Ollama available, using model: {self.client.config.model}")
            else:
                print("✗ Ollama not available, using rule-based fallback")
    
    def narrate(self, sql_query: str) -> str:
        """Convert SQL to human-readable narrative."""
        self._check_ollama()
        
        components = extract_sql_components(sql_query)
        narratives = []
        
        for label, sql_fragment in components:
            if self._use_ollama:
                try:
                    explanation = self._narrate_with_ollama(sql_fragment)
                except Exception as e:
                    explanation = self._narrate_rule_based(sql_fragment)
            else:
                explanation = self._narrate_rule_based(sql_fragment)
            
            narratives.append(f"{label}: {explanation}")
        
        return "\n".join(narratives)
    
    def _narrate_with_ollama(self, sql_fragment: str) -> str:
        """Use Ollama for natural language generation."""
        # Truncate if too long
        sql_fragment = sql_fragment[:MAX_SQL_LENGTH]
        
        prompt = USER_PROMPT_TEMPLATE.format(sql=sql_fragment)
        response = self.client.generate(prompt, system=SYSTEM_PROMPT)
        
        # Clean up response
        response = response.strip()
        # Remove common prefixes
        for prefix in ["This query ", "This SQL ", "The query ", "The SQL "]:
            if response.startswith(prefix):
                response = response[len(prefix):]
                response = response[0].upper() + response[1:] if response else response
                break
        
        return response
    
    def _narrate_rule_based(self, sql_fragment: str) -> str:
        """Fallback rule-based narrator."""
        try:
            parsed = sqlglot.parse_one(sql_fragment, error_level=ErrorLevel.IGNORE)
            parts = []
            
            if isinstance(parsed, exp.Select):
                # Tables
                tables = [t.name for t in parsed.find_all(exp.Table)]
                if tables:
                    parts.append(f"queries {', '.join(tables[:3])}")
                
                # Aggregations
                aggs = []
                if list(parsed.find_all(exp.Count)):
                    aggs.append("counts")
                if list(parsed.find_all(exp.Sum)):
                    aggs.append("sums")
                if list(parsed.find_all(exp.Avg)):
                    aggs.append("averages")
                if aggs:
                    parts.append(f"calculates {', '.join(aggs)}")
                
                # Group by
                if parsed.find(exp.Group):
                    parts.append("groups results")
                
                # Filters
                if parsed.find(exp.Where):
                    parts.append("filters by conditions")
                
                # Joins
                joins = list(parsed.find_all(exp.Join))
                if joins:
                    parts.append(f"joins {len(joins)} table(s)")
                
                # Order/Limit
                if parsed.find(exp.Order):
                    parts.append("sorts results")
                if parsed.find(exp.Limit):
                    parts.append("limits output")
            
            return ", ".join(parts) if parts else "performs a database operation"
        
        except Exception:
            return "performs a database operation"


# ========================
# Convenience function
# ========================
_narrator = None

def narrate_sql(sql_query: str, ollama_model: str = None) -> str:
    """
    Convert SQL query to human-readable narrative.
    
    Args:
        sql_query: SQL string to explain
        ollama_model: Optional model override (e.g., "mistral:7b")
    
    Returns:
        Human-readable explanation
    """
    global _narrator
    
    if _narrator is None or (ollama_model and _narrator.client.config.model != ollama_model):
        config = OllamaConfig(model=ollama_model) if ollama_model else None
        _narrator = SQLNarrator(config)
    
    return _narrator.narrate(sql_query)


# ========================
# Test
# ========================
if __name__ == "__main__":
    test_queries = [
        """
        WITH volume_calc AS (
            SELECT token, sum(amount) as total FROM trades GROUP BY 1
        )
        SELECT * FROM volume_calc WHERE total > 1000
        """,
        """
        SELECT u.name, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2024-01-01'
        GROUP BY u.name
        HAVING COUNT(o.id) > 5
        ORDER BY order_count DESC
        LIMIT 10
        """,
        """
        SELECT 
            date_trunc('day', timestamp) as day,
            token_address,
            sum(amount) as volume
        FROM dex_trades
        WHERE timestamp > now() - interval '7 days'
        GROUP BY 1, 2
        ORDER BY volume DESC
        """
    ]
    
    print("=" * 60)
    print("SQL Narrator (Ollama)")
    print("=" * 60)
    
    for i, sql in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        print(narrate_sql(sql))
