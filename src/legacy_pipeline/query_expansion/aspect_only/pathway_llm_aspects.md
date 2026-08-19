# Aspect Extraction: LLM Pathway

## Objective
To strictly parse a complex user query into a weighted list of orthogonal core aspects (entities, constraints, actions) using a Large Language Model. **This pathway performs zero keyword expansion.** It only identifies the concepts explicitly present in the query.

## Core Principles
1. **Constraint Completeness:** Never drop short acronyms (e.g., "AI", "API") or negations.
2. **Anti-Hallucination:** Never introduce concepts, product categories, or generic terms that are not explicitly stated or directly implied by the query. (e.g., Do not extract "Pricing" if the query only asks about "Features").
3. **Orthogonality:** Break the query down into mutually exclusive parts to maximize retrieval coverage. Do not extract overlapping aspects (e.g., "Adobe" and "Adobe Digital Experience" should just be "Adobe Digital Experience").

---

## Step-by-Step Flow

### Step 1: Prompt Construction
The orchestrator dynamically builds a strict system prompt.

**System Prompt Constraints:**
- "You are an expert query parser. Your goal is to break the user's query into a list of orthogonal aspects."
- "You must extract EXACTLY what is in the query. Do not add related concepts. Do not expand the terms."
- "Identify the core entities, actions, and constraints."
- "Combine tightly coupled modifiers (e.g., 'Third-party AI models' is one aspect, not 'Third-party' and 'AI models')."

### Step 2: LLM Inference
The prompt is sent to the LLM (e.g., Qwen 3.5 2B via `llama.cpp`).
- **Temperature:** `0.0` (Strict determinism).
- **Format:** Enforced JSON schema.

### Step 3: Schema Enforcement & Output
The LLM must return a JSON object strictly matching this schema:

```json
{
  "aspects": [
    {
      "name": "string (the extracted aspect)",
      "weight": "float (0.0 to 1.0, representing the importance of this constraint)"
    }
  ]
}
```

### Weighting Logic (Instructed to LLM)
- **1.0 (Critical):** Core entities or hard constraints (e.g., "Digital Experience", "AI").
- **0.5 - 0.9 (Secondary):** Actions or soft modifiers (e.g., "integrate", "strategy").

## Example Input / Output

**Input Query:** "How does Adobe integrate AI into its Digital Experience segment?"

**Expected JSON Output:**
```json
{
  "aspects": [
    {
      "name": "Digital Experience segment",
      "weight": 1.0
    },
    {
      "name": "AI integration",
      "weight": 1.0
    }
  ]
}
```
*(Notice "Adobe" is omitted as redundant context, and no generic product names are hallucinated).*
