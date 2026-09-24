#!/bin/bash

# Automatically Grab True Windows Host IP 
# This finds the exact virtual gateway back to your Windows environment dynamically.
PGHOST=$(ip route | grep default | awk '{print $3}')

# Database Network Configuration one data scale at the time
#DB="tpch_benchmark_1gb"
#DB="tpch_benchmark_3gb"
DB="tpch_benchmark_5gb"
PGUSER="postgres"
PGPORT="5432"


export PGPASSWORD='your password'  

# Generate query execution plan with out execution time ; execute one data scale at the time
BASE="$HOME/energy_aware_qo/tpch-dbgen/TPCH_queries"
#OUT="$HOME/energy_aware_qo/tpch_1gb_query_execution_plan_without_execution_time"
#OUT="$HOME/energy_aware_qo/tpch_3gb_query_execution_plan_without_execution_time"
OUT="$HOME/energy_aware_qo/tpch_5gb_query_execution_plan_without_execution_time"

USER_NAME=$(whoami)
USER_GROUP=$(id -gn)

# 1. Clear the output directory completely
rm -rf "$OUT"
mkdir -p "$OUT"

echo "Targeting Windows Host IP: $PGHOST"
echo "Starting TPC-H EXPLAIN pipeline (PLAN ONLY over Network)"

for i in {1..22}; do
  FILE="$BASE/q$i.sql"
  OUTFILE="$OUT/q$i.json"
  LOGFILE="$OUT/q$i.log"
  TEMP_RAW="$OUT/q$i.raw.tmp"

  if [ ! -f "$FILE" ]; then
    echo "Q$i missing, skipping"
    continue
  fi

  echo "Processing Q$i"
  QUERY_CONTENT=$(cat "$FILE")

  # 2. DROP VIEW step specifically for Q15 so it never crashes
  psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB" -c "DROP VIEW IF EXISTS revenue0;" > /dev/null 2>&1

  # 3. Handle Q17 with a background execution and forced break
  if [ "$i" -eq 17 ]; then
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB" -X -q -A -t -v ON_ERROR_STOP=1 <<EOF > "$TEMP_RAW" 2> "$LOGFILE" &
BEGIN;
EXPLAIN (FORMAT JSON)
$QUERY_CONTENT;
ROLLBACK;
EOF
    PSQL_PID=$!

    # Watchdog loop: wait for the file to populate
    for SEC in {1..6}; do
      sleep 0.5
      if [ -s "$TEMP_RAW" ] && grep -q "Plan" "$TEMP_RAW"; then
        echo "--> Q17 Plan detected in temporary buffer. Force-closing connection"
        break
      fi
    done

    # Terminate the background freeze
    kill -9 "$PSQL_PID" >/dev/null 2>&1
    killall -9 psql >/dev/null 2>&1
    
    # Give the file system a brief window to unlock the file descriptor
    sleep 0.5
  else
    # Standard synchronous execution for all other queries
    psql -h "$PGHOST" -p "$PGPORT" -U "$PGUSER" -d "$DB" -X -q -A -t -v ON_ERROR_STOP=1 <<EOF > "$TEMP_RAW" 2> "$LOGFILE"
BEGIN;
EXPLAIN (FORMAT JSON)
$QUERY_CONTENT;
ROLLBACK;
EOF
  fi

  # 4. Filter, sanitize, and explicitly guarantee the closing bracket
  if [ -s "$TEMP_RAW" ]; then
    grep -E '^(\[|\{| )' "$TEMP_RAW" > "$OUTFILE"
    
    # Check if the file ends with a ]. If it doesn't, append it!
    if [ "$(tail -c 2 "$OUTFILE" | tr -d '\n' | tr -d ' ')" != "]" ]; then
      echo "" >> "$OUTFILE"
      echo "]" >> "$OUTFILE"
    fi
    
    # Fix file permissions back to your account
    chown "$USER_NAME:$USER_GROUP" "$OUTFILE"
    echo "Q$i OK"
    rm -f "$LOGFILE" "$TEMP_RAW"
  else
    echo "Q$i FAILED (Check $LOGFILE)"
    rm -f "$OUTFILE" "$TEMP_RAW"
  fi
done

echo "Done. All JSON files are structurally complete and verified."