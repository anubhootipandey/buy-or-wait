# Buy or Wait?

A deterministic, evidence-aware financial decision system that evaluates whether a purchase is safe for a user based on their financial history, upcoming commitments, recurring spending, payment options, and additional evidence such as messages or images.

The core design principle is:

> **AI interprets ambiguous evidence; Python owns the financial math, forecasting, safety checks, and final decision.**

This keeps financial decisions deterministic, testable, and explainable.

---

## What it does

For each purchase request, the system can determine whether the user should:

* pay in full now
* use an available installment plan
* make a partial payment
* reduce eligible spending and proceed
* wait until a safer date
* avoid the purchase when no safe option exists

A payment is considered safe only if the user's balance remains above their required minimum throughout the forecast period.

---

## Architecture

```text
Financial data + purchase request + evidence
                    │
                    ▼
             Data validation
                    │
                    ▼
        Financial state reconstruction
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Recurring detection    Evidence layer
   + 90-day forecast      AI / deterministic
          │                   │
          └─────────┬─────────┘
                    ▼
          Normalized financial facts
                    │
                    ▼
           Deterministic planner
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Payment   Spending    Wait
        plans     changes
          │         │         │
          └─────────┼─────────┘
                    ▼
           Validate every option
                    │
                    ▼
             Final decision
```

### Safety boundary

AI is used for interpretation, not financial decision-making.

```text
Messy evidence
      │
      ▼
AI / evidence interpretation
      │
      ▼
Validated structured fact
      │
      ▼
Deterministic financial engine
      │
      ▼
Forecast → Plan → Validate → Decision
```

The model does **not** perform balance arithmetic, determine affordability, invent financial amounts, or choose the final payment plan.

---

## Project structure

```text
code/
├── data/
│   ├── models.py
│   ├── loader.py
│   ├── indexes.py
│   ├── currency.py
│   └── validate.py
│
├── engine/
│   ├── state.py
│   ├── reconciliation.py
│   ├── recurrence.py
│   ├── forecast.py
│   └── evidence_integration.py
│
├── planner/
│   ├── models.py
│   ├── affordability.py
│   ├── payment_plans.py
│   ├── spending_changes.py
│   └── ranking.py
│
├── evidence/
│   ├── models.py
│   ├── deterministic.py
│   ├── schema.py
│   ├── prompts.py
│   ├── ai_client.py
│   ├── validation.py
│   ├── resolver.py
│   └── cache.py
│
├── tests/
├── main.py
└── final_submission.py
```

---

## Financial engine

The financial engine converts raw transaction data into a forecastable cash-flow model.

### Transaction handling

The system accounts for different transaction states instead of treating every event as cash:

* settled transactions are realized cash
* failed and cancelled transactions are excluded
* pending credits are not treated as confirmed income
* scheduled future events are included in the forecast
* duplicate pending charges are excluded
* unrealized investment valuations are non-cash
* settled investment sales remain cash events
* unresolved amounts remain unresolved instead of becoming zero

Foreign-currency transactions use exact-date exchange rates and `Decimal` arithmetic for money calculations.

### Recurring transactions

Recurring income and expenses are detected from historical transaction patterns using deterministic cadence rules.

The detector supports:

* weekly
* biweekly
* monthly
* interleaved recurring series

Known future transactions can replace generated recurring occurrences to avoid double-counting.

### Forecasting

Each request is evaluated against a **90-day cash-flow forecast**.

The forecast processes events chronologically and checks the minimum-balance requirement at every relevant checkpoint.

---

## Evidence layer

Some important financial information may only appear in messages or other evidence.

Examples:

* salary changes
* cancelled recurring payments
* temporary amount changes
* future financial commitments
* corrections to transaction information

The evidence pipeline is:

```text
Evidence
   ↓
Deterministic checks
   ↓
AI interpretation when necessary
   ↓
Structured output
   ↓
Validation
   ↓
NormalizedFact
   ↓
Financial engine
```

AI-generated facts are validated before they can affect the financial model.

If evidence cannot be safely associated with a specific transaction or recurring series, the system does not guess.

### Gemini

Gemini is used for ambiguous text evidence through structured output.

The final financial calculations remain deterministic.

Evidence can also be cached so the financial pipeline does not depend on a live model call every time it runs.

---

## Planning engine

The planner evaluates safe ways to complete a purchase.

It can consider:

* full payment
* partial payment
* available installment options
* waiting for a safer date
* eligible spending reductions

Every candidate is simulated against the forecast.

A candidate is rejected if it causes the balance to fall below the required minimum.

### Spending changes

Only eligible flexible expenses can be modified. Protected expenses such as housing, utilities, healthcare, insurance, and debt payments are excluded.

The planner searches a bounded set of spending-change combinations rather than allowing unlimited reductions.

### Ranking

Valid plans are ranked deterministically using the challenge's priority order:

1. meet the desired completion date
2. avoid spending changes
3. minimize total amount paid
4. start earlier
5. minimize number of payments
6. use the lower payment-option ID as the final tie-breaker

---

## Running the project

From the repository root:

### Validate the dataset

```bash
python code/main.py
```

### Forecast a request

```bash
python code/main.py forecast --request-id request_27
```

### Run the planner

```bash
python code/main.py plan --request-id request_27
```

### Run tests

```bash
python -m unittest discover -s code/tests -t code -v
```

### Generate the final output

```bash
python code/final_submission.py
```

The submission runner produces:

```text
output.csv
```

with one decision per purchase request.

---

## Testing

The project includes tests covering:

* data loading and validation
* currency conversion
* transaction reconciliation
* recurrence detection
* forecasting
* affordability
* payment plans
* spending changes
* plan ranking
* evidence validation
* evidence integration
* ambiguity handling
* conflict resolution

The challenge implementation was verified with **172 tests passing** and was run against all **250 purchase requests**.

---

## Key design decisions

### Deterministic financial logic

Financial decisions should be reproducible and inspectable, so the LLM is kept outside the core calculation path.

### Evidence must be grounded

An ambiguous message should not silently change a user's financial forecast. When the system cannot confidently identify the target transaction or recurring series, it leaves the evidence unresolved.

### Exact money arithmetic

`Decimal` is used instead of floating-point arithmetic for financial calculations.

### Explicit ranking

Payment plans use deterministic lexicographic ranking rather than a weighted score, preventing a lower-priority criterion from overriding a higher-priority one.

---

## Current limitations

This project started as a HackerRank challenge implementation, so the current version is a prototype rather than a complete production financial service.

### Evidence and vision

The challenge submission manually resolved the organizer-provided image amounts. It does **not** provide a general-purpose live vision/OCR pipeline for arbitrary financial images.

A production version should use a vision/OCR model to extract structured fields such as:

* amount
* currency
* date
* transaction type
* amount meaning

and then validate those fields before they enter the financial engine.

### Forecasting

The current recurrence and forecasting logic is deterministic and rule-based. It prioritizes explainability but does not model every possible real-world financial pattern.

### Production infrastructure

A production deployment would additionally need:

* authentication and authorization
* secure financial-data storage
* API security
* observability and monitoring
* model/API failure handling
* rate limiting
* audit logging
* human review for ambiguous evidence
* continuous evaluation
* stronger protection of sensitive financial data

### Planning assumptions

Payment-plan availability, fees, interest, credit limits, and merchant-specific rules are constrained by the supplied challenge data. A real financial product would need to integrate with actual payment providers and user-specific eligibility.

---

## Design philosophy

The system deliberately separates **understanding** from **decision-making**:

```text
AI
"What does this evidence mean?"
        ↓
Structured fact
        ↓
Python
"What does this mean financially?"
        ↓
Forecast
        ↓
Safe payment options
        ↓
Validated decision
```

The goal is not to make the entire system AI-driven.

The goal is to use AI where it is useful while keeping the financial decisions **deterministic, explainable, and testable**.
