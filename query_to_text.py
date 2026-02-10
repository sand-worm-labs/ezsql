import os
import torch
import sqlglot
from sqlglot import exp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sqlglot.errors import ErrorLevel

# Authentication
os.environ["HF_TOKEN"] = "hf_"

model_name = "mrm8488/t5-base-finetuned-wikiSQL-sql-to-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_block(text):
    """Translates a small chunk of SQL."""
    inputs = tokenizer(f"translate Sql to English: {text}", return_tensors="pt", max_length=1512, truncation=False)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=2000)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_full_narrative(sql_query):
    try:
        parsed = sqlglot.parse_one(sql_query, error_level=ErrorLevel.IGNORE)
        narrative = []
        
        # 1. Handle CTEs (The 'With' blocks)
        ctes = list(parsed.find_all(exp.CTE))
        for cte in ctes:
            alias = cte.alias
            # Get the SQL inside the CTE
            cte_sql = cte.this.sql()
            explanation = translate_block(cte_sql)
            narrative.append(f"Step '{alias}': {explanation}")
        
        # 2. Handle the final Select
        main_query = parsed.copy()
        # Remove the WITH part so we only translate the final SELECT
        main_query.set("with", None)
        final_explanation = translate_block(main_query.sql())
        narrative.append(f"Finally: {final_explanation}")
        
        return "\n".join(narrative)
    except Exception as e:
        return f"Could not parse query: {e}"

# --- Running your massive query ---


# --- Test ---
query = """
with
  deposits as (
    SELECT
      nft_id,
      sum(value) as value,
      max(locktime) as lockTimestamp,
      from_unixtime(CAST(max(locktime) as BIGINT)) as lockTime
    from
      (
        SELECT
          cast(value / 1e18 as DECIMAL) as value,
          tokenId as nft_id,
          locktime
        from
          thena_fi_bnb.VotingEscrow_evt_Deposit
        where
          evt_block_time <= cast('2026-01-08 00:00:00' as TIMESTAMP)
        UNION all
        SELECT
          (-1) * cast(value / 1e18 as DECIMAL) as value,
          tokenId as nft_id,
          tokenId as locktime
        from
          thena_fi_bnb.VotingEscrow_evt_Withdraw
        where
          evt_block_time <= cast('2026-01-08 00:00:00' as TIMESTAMP)
      ) x
    GROUP by
      1
  ),
  nft_transfers as (
    SELECT
      old_owner,
      new_owner,
      nft_id,
      evt_tx_hash
    from
      (
        SELECT
          "from" as old_owner,
          to as new_owner,
          tokenId as nft_id,
          evt_block_time,
          evt_tx_hash,
          row_number() over (
            partition by
              tokenId
            order by
              evt_block_time desc
          ) as row_num
        from
          thena_fi_bnb.VotingEscrow_evt_Transfer
        where
          evt_block_time <= cast('2026-01-08 00:00:00' as TIMESTAMP)
      ) x
    where
      row_num = 1
    order by
      nft_id asc
  )
  --10205
,
  sanity_check_transfers as (
    SELECT
      nft_id,
      COUNT(evt_tx_hash) as counter
    from
      nft_transfers
    where
      nft_id > cast(0 as uint256)
    GROUP by
      1
    order by
      2 desc,
      1
  )
  --SELECT * from sanity_check_transfers
  --10205
,
  sanity_check_deposits as (
    SELECT
      *
    from
      deposits
    order by
      nft_id asc
  )
  --SELECT * from sanity_check_deposits
,
  view as (
    SELECT
      owner_link,
      owner_link2,
      nft_id,
      value,
      lockTime,
      lockTimestamp,
      new_owner
    from
      (
        SELECT
          '<a href=https://debank.com/profile/' || cast(t.new_owner as varchar) || ' target=_blank>' || '👤 ' || substring(cast(t.new_owner as varchar), 1, 4) || '...' || substring(cast(t.new_owner as varchar), 39, 42) || '</a>' as owner_link,
          '<a href=https://bscscan.com/address/' || cast(t.new_owner as varchar) || ' target=_blank>' || '👤 ' || substring(cast(t.new_owner as varchar), 1, 4) || '...' || substring(cast(t.new_owner as varchar), 39, 42) || '</a>' as owner_link2,
          t.nft_id,
          d.value,
          lockTime,
          lockTimestamp,
          new_owner,
          if(sc.address is not null, 'SC', 'EOA') as account_type
        from
          nft_transfers t
          join deposits d on t.nft_id = d.nft_id
          left join (
            select distinct
              address
            from
              bnb.traces
            where
              "type" = 'create'
          ) sc on t.new_owner = sc.address
        where
          t.nft_id > cast(0 as uint256)
          and t.new_owner != 0x0000000000000000000000000000000000000000
      ) x
    where
      account_type = 'EOA'
    order by
      4 desc
  ),
  raw_list as (
    select
      owner_link,
      owner_link2,
      new_owner,
      nft_id,
      sum(value) as value,
      MAX(lockTime) as lockTime,
      MAX(lockTimestamp) as lockTimestamp
    from
      view
    where
      new_owner != 0xae1c38847fb90a13a2a1d7e5552ccd80c62c6508
    GROUP by
      1,
      2,
      3,
      4
  )
select
  row_num,
  owner_link,
  owner_link2,
  new_owner,
  nft_id,
  value,
  lockTime,
  lockTimestamp
from
  (
    select
      row_number() over (
        order by
          value desc
      ) as row_num,
      owner_link,
      owner_link2,
      new_owner,
      nft_id,
      value,
      lockTime,
      lockTimestamp
    from
      raw_list
    order by
      5 desc
  )

  -- and lockTimestamp >= cast(1735689600 as uint256)
  -- Locked for more than 2025
  -- Curent MAX Lock = 1753315200'
""" 
# (Shortened for display, but works for your long query too)

print(f"\n\n {get_full_narrative(query)}")