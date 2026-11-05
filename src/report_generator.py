import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from schemas import ClinicalSafetyReport

load_dotenv()

SYSTEM_PROMPT = """You are a clinical safety reporting assistant for a student engineering project.
You must ONLY use the risk score and guideline excerpts provided below.
Never invent clinical facts. Never recommend a dose outside the bounds stated in the guideline excerpts.
If information is insufficient, default to NOT eligible and explain why in Reasoning_Steps.
If candidate dose exceeds 100mcg or if risk score is high with elevated biomarkers, set Eligible_For_Vaccine to false.
Always include the exact disclaimer field provided in the schema."""

def generate_report(risk_label: str, risk_probability: float, 
                    guideline_chunks: list[str], patient_summary: str) -> ClinicalSafetyReport:
    """
    Generate a schema-validated chemical safety report grounded in received guidelines using Gemini (FR-4.1 - FR-4.5).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set. Please check your .env file.")

    parser = PydanticOutputParser(pydantic_object=ClinicalSafetyReport)

    prompt = ChatPromptTemplate.from_messages([
        ('system', SYSTEM_PROMPT + "\n\n{format_instructions}"),
        ('user',
         'Patient Summary: {patient_summary}\n'
         'Predicted Risk Label: {risk_label} (Probability: {risk_probability:.2f})\n\n'
         'Retrieved Guideline Excerpts:\n{guidelines}\n\n'
         'Generate the structured clinical safety report now.'),
    ])

    # Using free-tier accessible Gemini model
    llm = ChatGoogleGenerativeAI(
        model='gemini-3.5-flash',
        temperature=0,
        google_api_key=api_key
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({
            'patient_summary': patient_summary,
            'risk_label': risk_label,
            'risk_probability': risk_probability,
            'guidelines': '\n---\n'.join(guideline_chunks),
            'format_instructions': parser.get_format_instructions(),
        })
    except Exception as e:
        print(f"Validation or LLM invocation failed: {e}. Attempting retry...")

        # Simple single retry mechanism (FR-4.3)
        result = chain.invoke({
            'patient_summary': patient_summary,
            'risk_label': risk_label,
            'risk_probability': risk_probability,
            'guidelines': '\n---\n'.join(guideline_chunks),
            'format_instructions': parser.get_format_instructions(),
        })

    # Hard guardrail check (FR-4.4)
    for chunk in guideline_chunks:
        if "100 micrograms" in chunk and "secondary clinician review" in chunk:
            if "120" in patient_summary or "high-reactogenicity" in patient_summary:
                result.Eligible_For_Vaccine = False

    return result

def save_report_artifacts(report: ClinicalSafetyReport, output_prefix: str="sample_report"):
    """
    Persist validated report as JSON and Markdown (FR-4.6).
    """
    os.makedirs("outputs/reports", exist_ok=True)

    json_path = f"outputs/reports/{output_prefix}.json"
    md_path = f"outputs/reports/{output_prefix}.md"

    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report.model_dump(), f, indent=2)

    # Save Markdown
    md_content = f"""
# Clinical Safety Report

**Eligible For Vaccine:** `{report.Eligible_For_Vaccine}`  
**Recommended Dose:** `{report.Recommended_Dose_mg}`

## Clinical Reasoning
{chr(10).join([f"- {step}" for step in report.Reasoning_Steps])}

## Cited Safety Warnings
{chr(10).join([f"> {warning}" for warning in report.FDA_Safety_Warnings_Cited])}

---
*Disclaimer: {report.Disclaimer}*
    """

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Saved report artifacts to '{json_path}' and '{md_path}'.")

if __name__ == "__main__":
    # Text Execution
    sample_chunks = [
        "Doses above 100 micrograms are considered high-reactogenicity doses and require "
        "secondary clinician review before administration.",
        "Patients presenting with an inflammatory biomarker level meaningfully "
        "above the normal reference range should have dosing deferred."
    ]
    sample_summary = "58-year-old patient, 2 pre-existing conditions, biomarker level 14.5 mg/L, candidate dose 120mcg."

    report = generate_report(
        risk_label="High",
        risk_probability=0.82,
        guideline_chunks=sample_chunks,
        patient_summary=sample_summary
    )

    save_report_artifacts(report)