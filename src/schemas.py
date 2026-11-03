from pydantic import BaseModel, Field
from typing import List

class ClinicalSafetyReport(BaseModel):
    Eligible_For_Vaccine: bool = Field(
        ..., description="Whether the patient is eligible given predicted risk and guideline rules."
    )
    Recommended_Dose_mg: float = Field(
        ..., description="Recommended dose in micrograms/milligrams, strictly respecting retrieved guideline bounds."
    )
    Reasoning_Steps: List[str] = Field(
        ..., description="Ordered, human-readable reasoning steps explaining the clinical recommendation."
    )
    FDA_Safety_Warnings_Cited: List[str] = Field(
        ..., description="Exact regulatory guideline snippets or safety warnings the recommendation relies upon."
    )
    Disclaimer: str = Field(
        default="Student engineering project output - not real clinical guidance.",
        description="Mandatory project safety disclaimer present in every generated report."
    )