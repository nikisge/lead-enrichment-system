# Lead Enrichment System v4 - KI-basierte Optimierung

## Implementierungsplan

**Erstellt:** Januar 2026
**Ziel:** Smarter, KI-basierter Enrichment-Prozess mit besserer Kontakt-Erkennung
**Laufzeit-Ziel:** 1-3 Minuten pro Lead

---

## 1. Problemanalyse

### Identifizierte Fehler

| Problem | Beispiel | Ursache | Lösung |
|---------|----------|---------|--------|
| Falsche Namen | "Weitere Möglichkeiten zu helfen" bei DKMS | Regex-Extraktion aus HTML | KI-basierte Extraktion |
| Falsche E-Mail-Domain | `annakaiser@freewheel.com` bei Diakoneo | Keine Domain-Validierung | KI prüft kontextabhängig |
| Team-Seiten nicht gefunden | DKMS Team unter `/informieren/ueber-die-dkms/...` | Nur feste URL-Pfade | Google Discovery + KI |

### Aktueller Flow (Problematisch)
```
Job Posting → Regex-Parsing → Feste URLs → Regex-Extraktion → Schwache Validierung
```

### Neuer Flow (KI-basiert)
```
Job Posting → LLM-Parsing → Google Discovery → KI-Extraktion → KI-Validierung
```

---

## 2. Architektur-Änderungen

### 2.1 Neue Dateien

```
clients/
├── llm_client.py          # NEU: Unified LLM Client (OpenRouter)
├── ai_extractor.py        # NEU: KI-basierte Datenextraktion
├── ai_validator.py        # NEU: KI-basierte Validierung
├── team_discovery.py      # NEU: Google + KI Team Page Discovery
├── impressum.py           # MODIFIZIEREN: KI-Extraktion integrieren
├── linkedin_search.py     # MODIFIZIEREN: KI-Validierung integrieren
├── fullenrich.py          # UNVERÄNDERT
├── kaspr.py               # UNVERÄNDERT
└── ...

pipeline.py                # REFACTOREN: Neuer Flow mit KI
config.py                  # ERWEITERN: OpenRouter Key
```

### 2.2 Bestehende Dateien - Kompatibilität

| Datei | Änderung | Risiko |
|-------|----------|--------|
| `pipeline.py` | Refactoring | Mittel - Hauptlogik |
| `impressum.py` | Erweiterung | Niedrig - Additive Änderung |
| `linkedin_search.py` | Erweiterung | Niedrig - Additive Änderung |
| `config.py` | +1 Key | Sehr niedrig |
| `models.py` | Ggf. neue Models | Niedrig |
| `main.py` | UNVERÄNDERT | Kein Risiko |

---

## 3. Detaillierter Implementierungsplan

### Phase 1: Grundlagen (LLM Client)

#### 3.1 Config erweitern
**Datei:** `config.py`

```python
# Hinzufügen:
openrouter_api_key: str = ""
```

**Datei:** `.env`
```
OPENROUTER_API_KEY=sk-or-...
```

#### 3.2 LLM Client erstellen
**Datei:** `clients/llm_client.py` (NEU)

**Funktionen:**
- `LLMClient` Klasse mit OpenRouter-Integration
- Modell-Auswahl: fast (Gemini 3 Flash), balanced (Haiku 4.5), smart (Sonnet 4.5)
- Retry-Logik und Error-Handling
- Token-Counting für Kostenkontrolle
- Context-Limit-Prüfung

**Modelle:**
| Alias | Modell | Preis (1M tokens) | Einsatz |
|-------|--------|-------------------|---------|
| fast | google/gemini-3-flash | $0.50/$3 | Validierungen, einfache Analyse |
| balanced | anthropic/claude-haiku-4.5 | $1/$5 | Extraktion, komplexere Analyse |
| smart | anthropic/claude-sonnet-4.5 | $3/$15 | Planung, Sales Brief |

---

### Phase 2: KI-Extraktion

#### 3.3 AI Extractor erstellen
**Datei:** `clients/ai_extractor.py` (NEU)

**Funktionen:**

1. `extract_contacts_from_page(page_text, company_name, page_type)`
   - Input: Rohtext von Puppeteer/httpx
   - Output: Liste von Kontakten mit Name, Titel, E-Mail, Telefon
   - Modell: balanced (Haiku 4.5)
   - Max Input: 10.000 Zeichen (Context-Schutz)

2. `extract_impressum_data(page_text, company_name)`
   - Extrahiert: Geschäftsführer, Telefone, E-Mails, Adresse
   - Geschäftsführer werden als Kandidaten zurückgegeben!
   - Modell: fast (Gemini 3 Flash)

3. `extract_job_contact(page_text, company_name)`
   - Speziell für Job-Posting-Seiten
   - Sucht nach Ansprechpartner-Mustern
   - Modell: fast (Gemini 3 Flash)

---

### Phase 3: KI-Validierung

#### 3.4 AI Validator erstellen
**Datei:** `clients/ai_validator.py` (NEU)

**Funktionen:**

1. `validate_contact_name(name)`
   - Prüft: Echter Personenname vs. Überschrift/Platzhalter
   - Output: `{valid: bool, reason: str}`
   - Modell: fast

2. `validate_email_for_company(email, company_name, company_domain)`
   - Prüft: Gehört E-Mail zur Firma?
   - Kontextabhängig: Subdomains, Mutter/Tochter OK
   - Komplett andere Firma = NICHT OK
   - Output: `{valid: bool, reason: str}`
   - Modell: fast

3. `validate_and_rank_candidates(candidates, company_name, company_domain, job_category)`
   - Validiert alle Kandidaten
   - Rankt nach Relevanz: HR > Abteilungsleiter > GF > Sonstige
   - Filtert ungültige raus
   - Output: Sortierte Liste der validen Kandidaten
   - Modell: balanced

4. `validate_linkedin_match(linkedin_snippet, person_name, company_name)`
   - Prüft: Arbeitet Person AKTUELL bei der Firma?
   - Output: `{is_current: bool, confidence: float}`
   - Modell: fast

---

### Phase 4: Team Discovery

#### 3.5 Team Discovery erstellen
**Datei:** `clients/team_discovery.py` (NEU)

**Funktionen:**

1. `discover_team_pages(company_name, domain)`
   - Google-Suchen:
     - `"{company}" Team Geschäftsführung site:{domain}`
     - `"{company}" Ansprechpartner Mitarbeiter`
     - `"{company}" über uns Team`
   - KI analysiert Snippets
   - Gibt Top 2-3 vielversprechende URLs zurück
   - Modell: fast

2. `scrape_and_extract_team(urls, company_name)`
   - Scrapt URLs mit Puppeteer (max 50KB pro Seite)
   - KI-Extraktion der Kontakte
   - Deduplizierung
   - Modell: balanced

3. `fallback_linkedin_search(company_name, positions)`
   - FALLBACK wenn keine Team-Kontakte gefunden
   - Google: `"{company}" "{position}" site:linkedin.com/in`
   - Positionen: Geschäftsführer, HR Manager, Personalleiter
   - KI validiert ob Person aktuell dort arbeitet
   - Modell: fast

---

### Phase 5: Pipeline Refactoring

#### 3.6 Pipeline umbauen
**Datei:** `pipeline.py`

**Neuer Flow:**

```python
async def enrich_lead(payload, skip_paid_apis=False):
    """
    NEUER KI-BASIERTER ENRICHMENT FLOW
    """

    # ===== PHASE 1: INITIALE DATENSAMMLUNG =====

    # 1. Job Posting parsen (Claude Sonnet 4.5 - wie bisher)
    parsed = await parse_job_posting(payload)

    # 2. Domain finden falls nicht vorhanden
    if not parsed.company_domain:
        parsed.company_domain = await google_find_domain(parsed.company_name)

    # ===== PHASE 2: PARALLEL SCRAPING =====

    # Parallel ausführen:
    job_contact_task = scrape_job_url_with_ai(payload.url, parsed.company_name)
    impressum_task = scrape_impressum_with_ai(parsed.company_domain, parsed.company_name)
    team_discovery_task = discover_and_scrape_team(parsed.company_name, parsed.company_domain)

    job_contact, impressum_data, team_contacts = await asyncio.gather(
        job_contact_task,
        impressum_task,
        team_discovery_task
    )

    # ===== PHASE 3: KANDIDATEN SAMMELN =====

    all_candidates = []

    # Priorität 1: Kontakt aus Job Posting (beste Quelle)
    if job_contact and job_contact.name:
        all_candidates.append({
            "name": job_contact.name,
            "email": job_contact.email,
            "title": job_contact.title,
            "source": "job_posting",
            "priority": 100
        })

    # Priorität 2: Kontakt aus LLM-Parsing
    if parsed.contact_name:
        all_candidates.append({
            "name": parsed.contact_name,
            "email": parsed.contact_email,
            "source": "llm_parse",
            "priority": 90
        })

    # Priorität 3: Team-Seiten-Kontakte
    for contact in team_contacts:
        all_candidates.append({
            **contact,
            "source": "team_page",
            "priority": 70
        })

    # Priorität 4: Geschäftsführer aus Impressum
    for exec in impressum_data.executives:
        all_candidates.append({
            "name": exec.name,
            "title": exec.title,
            "source": "impressum",
            "priority": 50
        })

    # ===== PHASE 4: KI-VALIDIERUNG & RANKING =====

    validated_candidates = await validate_and_rank_candidates(
        candidates=all_candidates,
        company_name=parsed.company_name,
        company_domain=parsed.company_domain,
        job_category=payload.category
    )

    # Top 3 nehmen
    top_candidates = validated_candidates[:3]

    # ===== FALLBACK: LINKEDIN-SUCHE =====

    if not top_candidates:
        # Keine Kandidaten gefunden - LinkedIn Fallback
        top_candidates = await fallback_linkedin_search(
            company_name=parsed.company_name,
            positions=["Geschäftsführer", "HR Manager", "Personalleiter"]
        )

    # ===== PHASE 5: LINKEDIN PROFILE FINDEN =====

    for candidate in top_candidates:
        if not candidate.get("linkedin_url"):
            linkedin_url = await find_linkedin_with_ai_validation(
                name=candidate["name"],
                company=parsed.company_name,
                domain=parsed.company_domain
            )
            candidate["linkedin_url"] = linkedin_url

    # ===== PHASE 6: PHONE ENRICHMENT =====

    phone_result = None
    decision_maker = None

    for candidate in top_candidates:
        # FullEnrich (funktioniert ohne LinkedIn!)
        phone_result = await try_fullenrich(candidate, parsed)

        if not phone_result and candidate.get("linkedin_url"):
            # Kaspr (braucht LinkedIn)
            phone_result = await try_kaspr(candidate)

        if phone_result:
            decision_maker = candidate
            break

    # Falls kein Telefon, nimm ersten Kandidaten
    if not decision_maker and top_candidates:
        decision_maker = top_candidates[0]

    # ===== PHASE 7: COMPANY RESEARCH =====

    company_intel = await research_company_with_ai(
        company_name=parsed.company_name,
        domain=parsed.company_domain,
        job_description=payload.description
    )

    company_linkedin = await google_find_company_linkedin(
        parsed.company_name,
        parsed.company_domain
    )

    # ===== FINAL OUTPUT =====

    return EnrichmentResult(
        success=decision_maker is not None,
        company=company_info,
        company_intel=company_intel,
        decision_maker=decision_maker,
        phone=phone_result,
        emails=collected_emails,
        enrichment_path=enrichment_path
    )
```

---

## 4. Kostenübersicht

### LLM-Kosten pro Lead (geschätzt)

| Schritt | Modell | Input Tokens | Output Tokens | Kosten |
|---------|--------|--------------|---------------|--------|
| Job Parsing | Sonnet 4.5 | ~2000 | ~200 | $0.009 |
| Team Discovery (Snippets) | Gemini 3 Flash | ~1000 | ~100 | $0.001 |
| Team Extraktion (2 Seiten) | Haiku 4.5 | ~4000 | ~300 | $0.006 |
| Impressum Extraktion | Gemini 3 Flash | ~2000 | ~200 | $0.002 |
| Validierung (5 Kandidaten) | Gemini 3 Flash | ~1500 | ~300 | $0.002 |
| LinkedIn Validierung (3x) | Gemini 3 Flash | ~500 | ~100 | $0.001 |
| Company Research | Sonnet 4.5 | ~3000 | ~500 | $0.017 |
| **GESAMT** | | ~14.000 | ~1.700 | **~$0.04** |

**Pro Lead: ~4 Cent LLM-Kosten** (sehr günstig!)

### Vergleich alte vs. neue Kosten

| Service | Alte Kosten | Neue Kosten |
|---------|-------------|-------------|
| FullEnrich | 10 Credits | 10 Credits (unverändert) |
| Kaspr | 1 Credit | 1 Credit (unverändert) |
| LLM (gesamt) | ~$0.01 | ~$0.04 |
| **GESAMT** | ~$0.01 + Credits | ~$0.04 + Credits |

Der Mehrwert (bessere Kontakt-Erkennung, weniger Fehlschläge) rechtfertigt die +3 Cent.

---

## 5. Context-Limits & Sicherheit

### Token-Limits

| Modell | Context Limit | Unser Max Input |
|--------|---------------|-----------------|
| Gemini 3 Flash | 1M tokens | 10.000 Zeichen (~2.500 tokens) |
| Claude Haiku 4.5 | 200K tokens | 15.000 Zeichen (~3.750 tokens) |
| Claude Sonnet 4.5 | 200K (1M optional) | 20.000 Zeichen (~5.000 tokens) |

### Scraping-Limits

```python
# Maximale Größen
MAX_PAGE_SIZE = 100_000  # 100KB HTML
MAX_TEXT_EXTRACT = 50_000  # 50KB Text
MAX_LLM_INPUT = 10_000  # 10KB für LLM

# Bei Überschreitung: Truncate intelligent
def truncate_for_llm(text: str, max_chars: int = 10000) -> str:
    if len(text) <= max_chars:
        return text
    # Priorisiere Anfang und Ende (Impressum oft am Ende)
    return text[:max_chars//2] + "\n...[truncated]...\n" + text[-max_chars//2:]
```

---

## 6. Implementierungsreihenfolge

### Schritt 1: Config & LLM Client
1. `config.py` erweitern (OPENROUTER_API_KEY)
2. `clients/llm_client.py` erstellen
3. Testen mit einfachem Prompt

### Schritt 2: AI Extractor
1. `clients/ai_extractor.py` erstellen
2. Prompts optimieren
3. Unit Tests

### Schritt 3: AI Validator
1. `clients/ai_validator.py` erstellen
2. Validierungs-Prompts
3. Unit Tests

### Schritt 4: Team Discovery
1. `clients/team_discovery.py` erstellen
2. Google Integration
3. Fallback-Logik
4. Unit Tests

### Schritt 5: Pipeline Integration
1. `pipeline.py` refactoren
2. Parallele Ausführung
3. Error Handling
4. Integration Tests

### Schritt 6: Testing & Optimierung
1. End-to-End Tests mit echten Daten
2. DKMS-Fall testen
3. Diakoneo-Fall testen
4. Performance messen
5. Prompts optimieren

---

## 7. Rollback-Strategie

Falls Probleme auftreten:

1. **Feature Flag**: `USE_AI_EXTRACTION = True/False` in config
2. **Alte Funktionen behalten**: `_extract_team_members_regex()` als Fallback
3. **Schrittweise Aktivierung**: Erst Validierung, dann Extraktion, dann Discovery

---

## 8. Erfolgs-Metriken

| Metrik | Aktuell | Ziel |
|--------|---------|------|
| Gültige Decision Maker | ~80% | >95% |
| E-Mail passt zu Firma | ~85% | >98% |
| Durchlaufzeit | ~3-5 min | <2 min |
| LLM-Kosten pro Lead | - | <$0.05 |
| Telefon-Erfolgsrate | ~60% | >70% |

---

## 9. Nächste Schritte

- [ ] `.env` um `OPENROUTER_API_KEY` erweitern
- [ ] `config.py` anpassen
- [ ] `clients/llm_client.py` implementieren
- [ ] `clients/ai_extractor.py` implementieren
- [ ] `clients/ai_validator.py` implementieren
- [ ] `clients/team_discovery.py` implementieren
- [ ] `pipeline.py` refactoren
- [ ] Tests durchführen
- [ ] Performance optimieren

---

**Status:** Bereit zur Implementierung
