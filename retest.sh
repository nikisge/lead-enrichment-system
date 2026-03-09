#!/bin/bash
# Serper Places+Search Re-Test Script
# Testet Domain-Discovery im Test-Modus (keine bezahlten APIs)
#
# Usage:
#   ./retest.sh                    # Alle 6 Test-Firmen
#   ./retest.sh "Firmenname"       # Einzelne Firma
#   ./retest.sh recent             # Letzte 20 Runs aus Dashboard re-testen

SERVER="http://localhost:8000"
RESULTS_FILE="/tmp/retest_results_$(date +%Y%m%d_%H%M%S).json"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

test_company() {
    local company="$1"
    local location="$2"
    local description="${3:-Stellenanzeige für $company}"

    echo -e "\n${YELLOW}━━━ Testing: $company ━━━${NC}"

    result=$(curl -s -X POST "$SERVER/webhook/enrich/test" \
        -H "Content-Type: application/json" \
        -d "{
            \"company\": \"$company\",
            \"description\": \"$description\",
            \"title\": \"Mitarbeiter\",
            \"id\": \"retest-$(date +%s)-$RANDOM\",
            \"location\": \"$location\"
        }" 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$result" ]; then
        echo -e "${RED}  ERROR: Request failed${NC}"
        return
    fi

    # Extract key fields
    domain=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('company',{}).get('domain','—'))" 2>/dev/null)
    phone=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('company',{}).get('phone','—'))" 2>/dev/null)
    address=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('company',{}).get('address','—') or '—')" 2>/dev/null)
    path=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(' → '.join(d.get('enrichment_path',[])))" 2>/dev/null)
    dm=$(echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); dm=d.get('decision_maker'); print(dm.get('name','—') if dm else '—')" 2>/dev/null)

    # Color-code results
    if [ "$domain" != "—" ] && [ "$domain" != "None" ]; then
        echo -e "  Domain:  ${GREEN}$domain${NC}"
    else
        echo -e "  Domain:  ${RED}nicht gefunden${NC}"
    fi

    if [ "$phone" != "—" ] && [ "$phone" != "None" ]; then
        echo -e "  Phone:   ${GREEN}$phone${NC}"
    else
        echo -e "  Phone:   ${RED}—${NC}"
    fi

    echo -e "  Address: $address"
    echo -e "  DM:      $dm"
    echo -e "  Path:    $path"

    # Append to results file
    echo "$result" | python3 -c "
import sys, json
d = json.load(sys.stdin)
entry = {
    'company': '$company',
    'domain': d.get('company',{}).get('domain'),
    'phone': d.get('company',{}).get('phone'),
    'address': d.get('company',{}).get('address'),
    'dm': d.get('decision_maker',{}).get('name') if d.get('decision_maker') else None,
    'path': d.get('enrichment_path',[])
}
print(json.dumps(entry, ensure_ascii=False))
" >> "$RESULTS_FILE" 2>/dev/null
}

# Mode: re-test recent runs from dashboard
if [ "$1" = "recent" ]; then
    echo -e "${YELLOW}Fetching last 20 runs from dashboard...${NC}"
    recent=$(curl -s "$SERVER/dashboard/recent?n=20" 2>/dev/null)

    if [ -z "$recent" ] || [ "$recent" = "[]" ]; then
        echo -e "${RED}No recent runs found${NC}"
        exit 1
    fi

    # Extract unique companies
    companies=$(echo "$recent" | python3 -c "
import sys, json
runs = json.load(sys.stdin)
seen = set()
for r in runs:
    name = r.get('company','')
    if name and name not in seen:
        seen.add(name)
        print(name)
" 2>/dev/null)

    count=$(echo "$companies" | wc -l | tr -d ' ')
    echo -e "Found ${GREEN}$count${NC} unique companies to re-test\n"

    while IFS= read -r company; do
        test_company "$company" "" ""
    done <<< "$companies"

# Mode: single company
elif [ -n "$1" ]; then
    test_company "$1" "${2:-}" ""

# Mode: default 6 test companies
else
    echo -e "${YELLOW}=== Serper Places+Search Re-Test ===${NC}"
    echo "Server: $SERVER"
    echo "Results: $RESULTS_FILE"

    test_company "089 Immobilienmanagement GmbH" "München"
    test_company "Regierung von Oberbayern" "München"
    test_company "Boston Consulting Group" "München"
    test_company "HAPEKO Hanseatisches Personalkontor" "Hamburg"
    test_company "HH Immobilien" "Hamburg"
    test_company "MLP Finanzberatung" "Heidelberg"
fi

echo -e "\n${YELLOW}━━━ Summary ━━━${NC}"
if [ -f "$RESULTS_FILE" ]; then
    total=$(wc -l < "$RESULTS_FILE" | tr -d ' ')
    domains=$(grep -c '"domain": "[^n]' "$RESULTS_FILE" 2>/dev/null || echo 0)
    phones=$(grep -c '"phone": "[^n]' "$RESULTS_FILE" 2>/dev/null || echo 0)
    echo -e "  Total:   $total"
    echo -e "  Domains: ${GREEN}$domains${NC}/$total"
    echo -e "  Phones:  ${GREEN}$phones${NC}/$total"
    echo -e "  Results: $RESULTS_FILE"
fi
