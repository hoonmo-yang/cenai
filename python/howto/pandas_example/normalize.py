from pydantic import BaseModel, Field
import pandas as pd

source = {
    "modality": "CT",
    "key_findings": "이것은 연습입니다.",
    "diagnostic_confidence": "Proven",
    "differential_diagnosis": "없음",
    "primary_tumor": {
        "number": "Single",
        "location": "Tail",
        "size": 3,
    },
    "tumor_vascular_invasion": {
        "artery": {
            "sma": "Encasement",
            "celiac_artery": "Abutment",
            "common_hepatic_artery": "Abutment",
            "proper_hepatatic_artery": "Abutment",
        },
        "vein": {
            "poral_vein": "No",
            "smv": "Encasement",
            "other_veins": "No",
        },
    },
    "regional_ln_metastasis": "Intermediate",
    "adjacent_organ_invasion": {
        "presence": "Probable",
        "location": "Liver",
    },
    "distant_metastasis": {
        "presence": "No",
        "location": "No",
    },
    "resetability": "Metastatic",
}

df = pd.json_normalize(source, sep="__")

x = df.to_dict(orient="records")

print(x)