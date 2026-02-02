# Troubleshooting Multi-Hop RAG

## Use Cases

- IT system fault diagnosis
- Equipment repair guidance
- Technical support tickets
- Production line problem solving

## Why Multi-Hop is Needed

Troubleshooting typically requires:
1. **Symptom Recognition**: User description → Match known fault patterns
2. **Root Cause Analysis**: Symptom → Possible causes → Root cause
3. **Solution Retrieval**: Based on cause → Find resolution steps
4. **Verification**: After resolution → Verify if fixed

A single retrieval cannot complete this chain.

## Knowledge Base Structure

```python
# Fault Pattern Database
fault_schema = client.create_schema()
fault_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
fault_schema.add_field("symptom", DataType.VARCHAR, max_length=65535)     # Symptom description
fault_schema.add_field("symptom_embedding", DataType.FLOAT_VECTOR, dim=1536)
fault_schema.add_field("fault_code", DataType.VARCHAR, max_length=32)     # Fault code
fault_schema.add_field("fault_name", DataType.VARCHAR, max_length=256)
fault_schema.add_field("severity", DataType.VARCHAR, max_length=16)       # critical/high/medium/low
schema.add_field("system", DataType.VARCHAR, max_length=64)               # System
schema.add_field("component", DataType.VARCHAR, max_length=64)            # Component

# Cause Database
cause_schema = client.create_schema()
cause_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
cause_schema.add_field("fault_id", DataType.VARCHAR, max_length=64)       # Related fault
cause_schema.add_field("cause_description", DataType.VARCHAR, max_length=65535)
cause_schema.add_field("cause_embedding", DataType.FLOAT_VECTOR, dim=1536)
cause_schema.add_field("probability", DataType.FLOAT)                      # Occurrence probability
cause_schema.add_field("is_root_cause", DataType.BOOL)                     # Is root cause

# Solution Database
solution_schema = client.create_schema()
solution_schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
solution_schema.add_field("cause_id", DataType.VARCHAR, max_length=64)    # Related cause
solution_schema.add_field("steps", DataType.VARCHAR, max_length=65535)    # Resolution steps
solution_schema.add_field("steps_embedding", DataType.FLOAT_VECTOR, dim=1536)
solution_schema.add_field("difficulty", DataType.VARCHAR, max_length=16)  # easy/medium/hard
solution_schema.add_field("estimated_time", DataType.INT32)               # Estimated time (minutes)
solution_schema.add_field("requires_expert", DataType.BOOL)               # Requires expert
```

## Implementation

```python
from pymilvus import MilvusClient
from openai import OpenAI

class TroubleshootingRAG:
    def __init__(self, uri: str = "./milvus.db"):
        self.client = MilvusClient(uri=uri)
        self.openai = OpenAI()

    def _embed(self, text: str) -> list:
        """Generate embedding using OpenAI API"""
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def diagnose(self, symptom_description: str, system: str = None) -> dict:
        """Complete diagnosis workflow"""
        result = {
            "symptom": symptom_description,
            "matched_faults": [],
            "possible_causes": [],
            "recommended_solutions": [],
            "diagnosis_path": []
        }

        # Hop 1: Symptom → Fault patterns
        faults = self._match_faults(symptom_description, system)
        result["matched_faults"] = faults
        result["diagnosis_path"].append({
            "step": 1,
            "action": "Symptom matching",
            "found": len(faults)
        })

        if not faults:
            return result

        # Hop 2: Fault → Causes
        all_causes = []
        for fault in faults[:3]:  # Take top 3 matching faults
            causes = self._find_causes(fault["id"])
            for c in causes:
                c["fault_id"] = fault["id"]
                c["fault_name"] = fault["fault_name"]
            all_causes.extend(causes)

        # Sort by probability
        all_causes.sort(key=lambda x: x.get("probability", 0), reverse=True)
        result["possible_causes"] = all_causes
        result["diagnosis_path"].append({
            "step": 2,
            "action": "Cause analysis",
            "found": len(all_causes)
        })

        # Hop 3: Cause → Solutions
        for cause in all_causes[:5]:  # Take top 5 most likely causes
            solutions = self._find_solutions(cause["id"])
            for s in solutions:
                s["cause_id"] = cause["id"]
                s["cause_description"] = cause["cause_description"]
            result["recommended_solutions"].extend(solutions)

        # Sort by difficulty and time (easier first)
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        result["recommended_solutions"].sort(
            key=lambda x: (difficulty_order.get(x.get("difficulty", "medium"), 1),
                          x.get("estimated_time", 60))
        )

        result["diagnosis_path"].append({
            "step": 3,
            "action": "Solution retrieval",
            "found": len(result["recommended_solutions"])
        })

        # Generate diagnosis summary with LLM
        result["summary"] = self._generate_summary(result)

        return result

    def _match_faults(self, symptom: str, system: str = None) -> list:
        """Match fault patterns"""
        embedding = self._embed(symptom)

        filter_expr = ""
        if system:
            filter_expr = f'system == "{system}"'

        results = self.client.search(
            collection_name="faults",
            data=[embedding],
            filter=filter_expr,
            limit=5,
            output_fields=["fault_code", "fault_name", "symptom", "severity", "system"]
        )

        return [{
            "id": r["id"],
            "fault_code": r["entity"]["fault_code"],
            "fault_name": r["entity"]["fault_name"],
            "symptom": r["entity"]["symptom"],
            "severity": r["entity"]["severity"],
            "match_score": r["distance"]
        } for r in results[0]]

    def _find_causes(self, fault_id: str) -> list:
        """Find fault causes"""
        results = self.client.query(
            collection_name="causes",
            filter=f'fault_id == "{fault_id}"',
            output_fields=["cause_description", "probability", "is_root_cause"],
            limit=10
        )

        # Sort by probability
        results.sort(key=lambda x: x.get("probability", 0), reverse=True)
        return results

    def _find_solutions(self, cause_id: str) -> list:
        """Find solutions"""
        results = self.client.query(
            collection_name="solutions",
            filter=f'cause_id == "{cause_id}"',
            output_fields=["steps", "difficulty", "estimated_time", "requires_expert"],
            limit=5
        )
        return results

    def _generate_summary(self, diagnosis: dict) -> str:
        """Generate diagnosis summary"""
        prompt = f"""Based on the following diagnosis results, generate a concise fault diagnosis report.

Symptom description: {diagnosis["symptom"]}

Matched fault patterns:
{chr(10).join([f"- {f['fault_name']} ({f['severity']})" for f in diagnosis["matched_faults"][:3]])}

Possible causes (sorted by probability):
{chr(10).join([f"- {c['cause_description']} (probability: {c.get('probability', 'N/A')})" for c in diagnosis["possible_causes"][:5]])}

Recommended solutions:
{chr(10).join([f"- {s['steps'][:100]}... (difficulty: {s.get('difficulty', 'N/A')}, estimated: {s.get('estimated_time', 'N/A')} min)" for s in diagnosis["recommended_solutions"][:3]])}

Please generate:
1. Most likely fault assessment
2. Recommended troubleshooting order
3. Important notes

Report:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    def interactive_diagnose(self, initial_symptom: str) -> None:
        """Interactive diagnosis"""
        print(f"Starting diagnosis: {initial_symptom}\n")

        # Initial diagnosis
        result = self.diagnose(initial_symptom)

        print("=== Initial Diagnosis Results ===")
        print(f"\nMatched faults:")
        for f in result["matched_faults"][:3]:
            print(f"  - [{f['fault_code']}] {f['fault_name']} (severity: {f['severity']})")

        print(f"\nPossible causes:")
        for c in result["possible_causes"][:5]:
            print(f"  - {c['cause_description']}")

        # Ask for more information
        print("\n=== More Information Needed ===")
        follow_up = self._generate_follow_up_questions(result)
        print(follow_up)

    def _generate_follow_up_questions(self, diagnosis: dict) -> str:
        """Generate follow-up questions"""
        prompt = f"""Based on the following diagnosis results, generate 3-5 follow-up questions to narrow down the fault scope.

Matched faults: {[f['fault_name'] for f in diagnosis['matched_faults'][:3]]}
Possible causes: {[c['cause_description'] for c in diagnosis['possible_causes'][:5]]}

Questions should:
1. Help distinguish between different fault possibilities
2. Confirm or rule out certain causes
3. Be concise and clear, easy for users to answer

Follow-up questions:"""

        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content
```

## Example

```python
rag = TroubleshootingRAG()

# Diagnose
result = rag.diagnose(
    "Server response is slow, CPU usage consistently above 90%",
    system="Linux Server"
)

print("=== Diagnosis Report ===")
print(f"\nSymptom: {result['symptom']}")

print("\n[Possible Faults]")
for f in result["matched_faults"]:
    print(f"  [{f['fault_code']}] {f['fault_name']}")
    print(f"      Severity: {f['severity']}, Match score: {f['match_score']:.2f}")

print("\n[Possible Causes]")
for c in result["possible_causes"][:5]:
    prob = c.get('probability', 0)
    print(f"  - {c['cause_description']} (probability: {prob:.0%})")

print("\n[Recommended Solutions]")
for s in result["recommended_solutions"][:3]:
    print(f"  [{s.get('difficulty', 'N/A')}] {s['steps'][:80]}...")
    print(f"      Estimated time: {s.get('estimated_time', 'N/A')} minutes")

print("\n[Diagnosis Summary]")
print(result["summary"])
```

## Knowledge Base Maintenance

```python
def add_fault_case(self, symptom: str, fault_code: str, fault_name: str,
                   causes: list, solutions: list):
    """Add fault case"""
    import uuid

    # Add fault pattern
    fault_id = str(uuid.uuid4())
    self.client.insert(
        collection_name="faults",
        data=[{
            "id": fault_id,
            "symptom": symptom,
            "symptom_embedding": self._embed(symptom).tolist(),
            "fault_code": fault_code,
            "fault_name": fault_name,
            "severity": "medium",
            "system": "",
            "component": ""
        }]
    )

    # Add causes
    for cause in causes:
        cause_id = str(uuid.uuid4())
        self.client.insert(
            collection_name="causes",
            data=[{
                "id": cause_id,
                "fault_id": fault_id,
                "cause_description": cause["description"],
                "cause_embedding": self._embed(cause["description"]).tolist(),
                "probability": cause.get("probability", 0.5),
                "is_root_cause": cause.get("is_root", False)
            }]
        )

        # Add solutions for this cause
        for sol in cause.get("solutions", []):
            self.client.insert(
                collection_name="solutions",
                data=[{
                    "id": str(uuid.uuid4()),
                    "cause_id": cause_id,
                    "steps": sol["steps"],
                    "steps_embedding": self._embed(sol["steps"]).tolist(),
                    "difficulty": sol.get("difficulty", "medium"),
                    "estimated_time": sol.get("time", 30),
                    "requires_expert": sol.get("expert", False)
                }]
            )
```
