from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, Field


class PDACResultClassify(BaseModel):
    class Config:
        title = "PDAC classification result"
        description = "predicted type of PDAC report by classification"

    type: str = Field(
        description="AI 분류기가 예측한 CT 판독문의 유형",
    )


class PDACReport(BaseModel):
    class Config:
        title = "PDAC report base class for configuration"
        description = "PDAC report base class for configuration"

        use_enum_values = True


class PDACReportTemplate(PDACReport):
    class Config:
        title = "PDAC report template"
        description = "Template for a report on pancreatic cancer diagnostic imaging"


    modality: ModalityEnum = Field(
        ...,
        description = "Modality used for diagnostic imaging",
    )

    summary: str = Field(
        description = """Summarizes the key findings in a report
                         on pancreatic cancer diagnostic imaging.""",
    )


class PDACReportTemplate1(PDACReportTemplate):
    class Config:
        title = "Type 1 template for a report on pancreatic cancer diagnostic imaging"

        description = """Type 1 template for a report on pancreatic cancer diagnostic imaging
                         (Name: initial diagnosis & staging)"""


    type: str = "1. initial diagnosis & staging"

    key_findings: KeyFindingsTemplate1 = Field(
        ...,
        description = "Template for key findings in a Type 1 report"
    )


class PDACReportTemplate2(PDACReportTemplate):
    class Config:
        title = "Type 2 template for a report on pancreatic cancer diagnostic imaging"

        description = """Type 2 template for a report on pancreatic cancer diagnostic imaging
                         (Name: follow-up for pancreatic cancer without curative resection)"""


    type: str = "2. follow-up for pancreatic cancer without curative resection"

    key_findings: KeyFindingsTemplate2 = Field(
        ...,
        description = "Template for key findings in a Type 2 report"
    )


class PDACReportTemplate3(PDACReportTemplate):
    class Config:
        title = "Type 3 template for a report on pancreatic cancer diagnostic imaging"

        description = """Type 3 template for a report on pancreatic cancer diagnostic imaging
                         (Name: Follow-up after curative resection of pancreatic cancer"""


    type: str = "3. follow-up after curative resection of pancreatic cancer"

    key_findings: KeyFindingsTemplate3 = Field(
        ...,
        description = "Template for key findings in a Type 3 report"
    )


class PDACReportTemplate4(PDACReportTemplate):
    class Config:
        title = "Type 4 template for a report on pancreatic cancer diagnostic imaging"

        description = """Type 4 template for a report on pancreatic cancer diagnostic imaging
                         (Name: Follow-up for tumor recurrence after curative resection"""


    type: str = "4. Follow-up for tumor recurrence after curative resection"

    key_findings: KeyFindingsTemplate4 = Field(
        ...,
        description = "Template for key findings in a Type 4 report"
    )


class PDACReportTemplateFail(PDACReport):
    class Config:
        title = "Missing template for a report on pancreatic cancer diagnostic imaging"

        description = """Error: missing template for a report on pancreatic cancer
                         diagnostic imaging"""

    type: str = "Not available"

    message: str = Field(
        default = "PDAC Report template not available",
        description = "Error message",
    )


class KeyFindingsTemplate1(PDACReport):
    class Config:
        title = "Template for key findings in a type 1 report"

        description = """Template for key findings in a type 1 report 
                         on pancreatic cancer diagnostic imaging
                         (Name: initial diagnosis & staging)"""


    diagnostic_confidence: ConfidenceDiagnosisEnum = Field(
        ...,
        description = """Level of certainity in the interpretation
                         of imaging findings.""",
    )

    differential_diagnosis: YesNoEnum = Field(
        ...,
        description = """Mention of differential diagnosis""",
    )

    primary_tumor: PrimaryTumor = Field(
        ...,
        description = """Findings on primary tumors detected
                         in the Type 1 diagnostic imaging report"""
    )

    tumor_vascular_invasion: TumorInvasionVascular = Field(
        ...,
        description = """Tumor vascular invasion""",
    )

    regional_ln_metastasis: PresenceEnum = Field(
        ...,
        description = """Presence of tumor metastasis in regional lymph nodes"""
    )

    adjacent_organ_invasion: TumorInvasionAdjacentOrgan = Field(
        ...,
        description = """Tumor invasion into adjacent organs""",
    )

    distant_metastasis: DistantMetastasis = Field(
        ...,
        description = """Presence of tumor metastasis in distant organs
                         or tissues beyond the primary site""",
    )

    resectability: TumorResectabilityEnum = Field(
        ...,
        description = """Surgical resectability of the tumor""",
    )


class KeyFindingsTemplate2(PDACReport):
    class Config:
        title = "Template for key findings in a type 2 report"

        description = """Template for key findings in a type 2 report 
                         on pancreatic cancer diagnostic imaging
                         (Name: follow-up for pancreatic cancer 
                         without curative resection)"""


    treatment: TreatmentEnum = Field(
        ...,
        description = """Type of treament provided to the patient""",
    )

    primary_tumor: StatusUndetecedEnum = Field(
        ...,
        description = """Status of the primary tumor""",
    )

    tumor_vascular_contact: StatusUndetecedEnum = Field(
        ...,
        description = """Status of tumor vascular contact""",
    )

    tumor_vascular_contact_specify: str = Field(
        ...,
        description = """Specification on status of "
                         the tumor vascular contact""",
    )

    regional_ln_metastasis: StatusUndetecedEnum = Field(
        ...,
        description = """Status of tumor metastasis in regional lymph nodes""",
    )

    distant_metastasis: StatusUndetecedEnum = Field(
        ...,
        description = """Status of tumor metastasis in distant organs
                         or tissues""",
    )

    new_lesion: NewLesion = Field(
        ...,
        description = """Presence of new lesion""",
    )

    overall_tumor_burden: StatusEnum = Field(
        ...,
        description = """Overall change in tumor burden""",
    )


class KeyFindingsTemplate3(PDACReport):
    class Config:
        title = "Template for key findings in a type 3 report"

        description = """Template for key findings in a type 3 report 
                         on pancreatic cancer diagnostic imaging
                         (Name: Follow-up after curative resection of
                         pancreatic cancer"""


    residual_pancreatic_tumor: PresenceEnum = Field(
        ...,
        description = """Presence of residual pancreatic tumor""",
    )

    residual_metastatic_lesion: PresenceEnum = Field(
        ...,
        description = """Presence of residual metastatic lesion""",
    )

    residual_metastatic_lesion_specify: str = Field(
        ...,
        description = """Specification on presence of residual 
                         metastatic lesion""",
    )

    tumor_recurrence: PresenceRecurrenceEnum = Field(
        ...,
        description = """Presence of tumor recurrence""",
    )

    tumor_recurrence_location: LocationTumorRecurrenceEnum = Field(
        ...,
        description = """Location of tumor recurrence""",
    )

    post_op_unusual_findings: PresenceUnusualFindingsEnum = Field(
        ...,
        description = """Presence of unusal findings after resection""",
    )

    post_op_unusual_findings_specify: str = Field(
        ...,
        description = """Specification on presence of unusal findings 
                         after resection""",
    )


class KeyFindingsTemplate4(PDACReport):
    class Config:
        title = "Template for key findings in a type 4 report"

        description = """Template for key findings in a type 4 report 
                         on pancreatic cancer diagnostic imaging
                         (Name: Follow-up for tumor recurrence
                         after curative resection"""


    treatment: TreatmentEnum = Field(
        ...,
        description = """Type of treament provided to the patient""",
    )

    tumor_recurrence_location: LocationTumorRecurrenceEnum = Field(
        ...,
        description = """Location of tumor recurrence""",
    )

    change_in_tumor_recurrence: StatusEnum = Field(
        ...,
        description = """Change in tumor recurrence""",
    )

    overall_tumor_burden: StatusEnum = Field(
        ...,
        description = """Overall change in tumor burden""",
    )


class PrimaryTumor(PDACReport):
    class Config:
        title = "Primary tumor"

        description = """Findings on primary tumors detected
                         in diagnostic imaging report"""


    number: NumberEnum = Field(
        ...,
        description = "Number of detected primary tumors" ,
    )

    location: LocationPrimaryTumorEnum = Field(
        ...,
        description = "Location of primary tumor",
    )

    size: float = Field(
        ...,
        description = """Largest diameter of the detected primary tumors (in centimeters)""",
    )


class TumorInvasionVascular(PDACReport):
    class Config:
        title = "Tumor vascular invasion"

        description = "Tumor vascular invasion"


    artery: TumorInvasionVascularArtery = Field(
        ...,
        description = "Tumor vascular invasion involving an artery" ,
    )

    vein: TumorInvasionVascularVein = Field(
        ...,
        description = "Tumor vascular invasion involving an vein",
    )


class TumorInvasionVascularArtery(PDACReport):
    class Config:
        title = "Tumor vascular invasion involving an artery"

        description = "Tumor vascular invasion involving an artery"


    sma: TumorInvasionVascularEnum = Field(
        ...,
        description = """Superior mesenteric artery (SMA): 
                         a major artery evaluated for tumor involvement""",
    )

    celiac_artery: TumorInvasionVascularEnum = Field(
        ...,
        description = """Celiac artery: 
                         a major artery evaluated for tumor involvement""",
    )

    common_hepatic_artery: TumorInvasionVascularEnum = Field(
        ...,
        description = """Common hepatic artery: 
                         a major artery evaluated for tumor involvement""",
    )

    proper_hepatic_artery: TumorInvasionVascularEnum = Field(
        ...,
        description = """Proper hepatic artery: 
                         a major artery evaluated for tumor involvement""",
    )

    first_jejunal_artery : TumorInvasionVascularEnum = Field(
        ...,
        description = """1st jejunal artery: 
                         a major artery evaluated for tumor involvement""",
    )

    aorta : TumorInvasionVascularEnum = Field(
        ...,
        description = """Aorta: 
                         a major artery evaluated for tumor involvement""",
    )

class TumorInvasionVascularVein(PDACReport):
    class Config:
        title = "Tumor vascular invasion involving a vein"

        description = "Tumor vascular invasion involving a vein"


    poral_vein: TumorInvasionVascularEnum = Field(
        ...,
        description = """Poral vein: 
                         a major vein evaluated for tumor involvement""",
    )

    smv: TumorInvasionVascularEnum = Field(
        ...,
        description = """Superior mesenteric vein (SMV): 
                         a major vein evaluated for tumor involvement""",
    )

    first_jejunal_vein : TumorInvasionVascularEnum = Field(
        ...,
        description = """1st jejunal vein: 
                         a major vein evaluated for tumor involvement""",
    )

    ivc : TumorInvasionVascularEnum = Field(
        ...,
        description = """Inferior Vena Cava (IVC): 
                         a major vein evaluated for tumor involvement""",
    )

    other_veins : TumorInvasionVascularEnum = Field(
        ...,
        description = """Other veins: 
                         major veins evaluated for tumor involvement""",
    )

    other_veins_specify : str = Field(
        ...,
        description = """List of other veins evaluated for tumor involvement"""
    )


class TumorInvasionAdjacentOrgan(PDACReport):
    class Config:
        title = "Tumor invasion into other organs"

        description = "Tumor invasion into other organs"


    presence: TumorInvasionAdjacentOrganEnum = Field(
        ...,
        description = """Adjacent organs evaluated for tumor involvement""",
    )

    location: str = Field(
        ...,
        description = """Location of adjacent organ invasion"""
    )


class DistantMetastasis(PDACReport):
    class Config:
        title = "distant metastasis"

        description = """Presence of tumor metastasis in distant organs
                         or tissues beyond the primary site"""

    presence: PresenceEnum = Field(
        ...,
        description = """presence of distant metastasis""",
    )

    location: str = Field(
        ...,
        description = """Location of distant metastasis"""
    )


class NewLesion(PDACReport):
    class Config:
        title = "new lesion"

        description = """Presence of new lesion"""


    location: str = Field(
        ...,
        description = """Location of new lesion""",
    )

    confidence_for_metastasis: ConfidenceMetastasisEnum = Field(
        ...,
        description = """Level of confidence in determining 
                         the presence of metastasis for new lesion""",
    )


class ConfidenceDiagnosisEnum(str, Enum):
    PROVEN = "Proven"
    PROBABLE = "Probable"
    SUSPICIOUS = "Suspicious"
    NOT_SURE = "Not sure"
    NOT_MENTIONED = "Not mentioned"


class ConfidenceMetastasisEnum(str, Enum):
    PROBABLE = "Probable"
    EQUIVOCAL = "Equvical"
    UNLIKELY = "Unlikely"


class LocationPrimaryTumorEnum(str, Enum):
    HEAD = "Head"
    BODY = "Body"
    TAIL = "Tail"
    NOT_MENTIONED = "Not mentioned"


class LocationTumorRecurrenceEnum(str, Enum):
    LOCAL = "Local"
    LN_METASTASIS = "LM metastasis"
    LIVER = "Liver"
    PERTIONEUM = "Peritoneum"
    OTHERS = "Others"
    NOT_MENTIONED = "Not mentioned"


class ModalityEnum(str, Enum):
    CT = "CT"
    MRI = "MRI"
    NOT_MENTIONED = "Not mentioned"


class NumberEnum(str, Enum):
    SINGLE = "Single"
    MULTIPLE = "Multiple"
    NOT_MENTIONED = "Not mentioned"


class PresenceEnum(str, Enum):
    NO = "No"
    INDETERMINATE = "Indeterminate"
    SUSPICIOUS_DEFINITE = "Suspicious/definite"
    NOT_MENTIONED = "Not mentioned"


class PresenceRecurrenceEnum(str, Enum):
    NO = "No"
    INDETERMINATE = "Indeterminate"
    PROBABLE_DEFINITE = "Probable/definite"
    NOT_MENTIONED = "Not mentioned"


class PresenceUnusualFindingsEnum(str, Enum):
    NO = "No"
    PRESENT = "Present"
    NOT_MENTIONED = "Not mentioned"


class StatusEnum(str, Enum):
    INCREASED = "Increased"
    STABLE = "Stable (no change)"
    DECREASED = "Decreased"
    NOT_MENTIONED = "Not mentioned"


class StatusUndetecedEnum(str, Enum):
    UNDETECTED = "Undetected"
    INCREASED = "Increased"
    STABLE = "Stable (no change)"
    DECREASED = "Decreased"
    NOT_MENTIONED = "Not mentioned"


class TreatmentEnum(str, Enum):
    NO_CONSERVATIVE = "No/conservative"
    CHEMOTHERAPY = "Chemotherapy"
    RADIATION = "Radiation"
    NOT_MENTIONED = "Not mentioned"


class TumorInvasionAdjacentOrganEnum(str, Enum):
    NO = "No"
    SUSPICIOUS = "Suspicious"
    PROBABLE = "Probable (abutment/invasion)"
    NOT_MENTIONED = "Not mentioned"


class TumorInvasionVascularEnum(str, Enum):
    NO = "No"
    ABUTMENT = "Abutment"
    ENCASEMENT = "Encasement"
    INVOLVEMENT_WITHOUT_SPECIFICATION = "Involvement without specification"
    NOT_MENTIONED = "Not mentioned"


class TumorResectabilityEnum(str, Enum):
    RESECTABLE = "Resectable"
    BORDERLINE = "Borderline resectable"
    LOCALLY_ADVANCED = "Locally advanced"
    METASTATIC = "Metastatic"
    NOT_MENTIONED = "Not mentioned"


class YesNoEnum(str, Enum):
    NO = "No"
    YES = "Yes"
