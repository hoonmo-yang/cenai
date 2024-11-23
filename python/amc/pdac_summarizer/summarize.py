import streamlit as st
import pandas as pd

content1 = "Technique:\n  조영제를 사용한 외부 MRI를 판독함. 외부에서 시행한 검사로 사용된 sequence의 구체적 내용을 알기 어려움.\n\nFindings:\n  Pancreaticoduodenal groove와 pancreatic head lower portion에 2 cm과 2.3 cm 크기의 ill-defined mass가 있으며 synchronous cancer 가능성이 높음 (CT SE 302 IM 171, 181). SMV와 닿아 있고 약간 좁아져 있음. 다른 major vessel invasion은 뚜렷하지 않음.\n  CBD invasion으로 bile duct가 심하게 늘어나 있음.\n  GB stone이 있음.\n  Liver에 metastasis 가능성이 높은 solid nodule이 몇 개 있음 (MR SE 1301 IM 855, 865, 872, 881).\n  Liver에 cyst들이 있음.\n  Hepatoduodenal ligament에 의미가 분명하지 않은 lymph node가 몇 개 있음.\n  Aorta 주변에 커진 lymph node 없음.\n  Both kidneys에 cyst들이 있음.\n  Uterine myomas 있음."
content2 = "Technique:\n  3T MR 기기 (Magnetom Skyra; Siemens Healthineers)를 사용함. Axial in & opposed T1WI, fat-suppressed axial T1WI, axial moderate T2WI, axial heavily T2WI, diffusion-weighted image, coronal T2WI, 2D MRCP, 3D MRCP를 획득함. Gd-DOTA (Dotarem®)를 주입하여, multiphase 3D gradient-echo fat-suppressed axial & coronal T1WI 획득함.\n\nTechnique:\n  Unenhanced 및 IV 조영제 주입한 후 동맥기와 정맥기의 two phase의 다중시기 촬영을 함.\n\nFindings:\n1. Pancreatic cancer:\n   - location: body/tail.\n   - maximum diameter: 1.3 cm.\n   - peripancreatic infiltration: present.\n   - major vascular invasion: absent.\n   - adjacent organ invasion: absent.\n\n2. Regional lymph node metastasis: absent.\n\n3. Distant metastasis: absent.\n\n4. Others: 1) a hepatic cyst in liver S8.\n           2) GB stones."

result_df = pd.DataFrame({
    "Content": [content1, content2],
    "Modality":	["CT", "CT"],
    "Key findings":	["", ""],
    "Diagnostic confidence": ["Probable", "Probable"],
    "Differntial diagnosis": ["Not mentioned", "Not mentioned"],				
    "Primary tumor__Number": ["Single", "Single"],
    "Primary tumor__Location": ["Body", "Body"],
    "Primary tumor__Size": ["2", "2"],
    "Tumor-vascular invasion__Artery__SMA": ["Abutment", "Abutment"],
    "Tumor-vascular invasion__Artery__Celiac artery": ["Encasement", "Abutment"],						
    "Tumor-vascular invasion__Artery__Common hepatic artery": ["Encasement", "Abutment"],						
    "Tumor-vascular invasion__Artery__Proper hepatic artery": ["Abutment", "Abutment"],						
    "Tumor-vascular invasion__Artery__1st Jejunal artery": ["Encasement", "Abutment"],						
    "Tumor-vascular invasion__Artery__Aorta": ["Encasement", "Abutment"],						
    "Tumor-vascular invasion__Vein__Poral vein": ["Encasement", "Abutment"],						
    "Tumor-vascular invasion__SMV": ["No", "No"],						
    "Tumor-vascular invasion__1st Jejunal vein": ["Encasement", "Encasement"],						
    "Tumor-vascular invasion__IVC": ["No", "No"],						
    "Tumor-vascular invasion__Other veins": ["No", "No"],						
    "Regional LN metastasis": ["indeterminate", "Indeterminte"],
    "Adjacent organ invasion__Status": ["indeterminate", "Indeterminte"],
    "Adjacent organ invasion__Presence": ["Suspicious", "Suspcicious"],
    "Adjacent organ invasion__Location": ["Liver", "Liver"],					
    "Distant metastasis__Status": ["indeterminate", "indetermintate"],
    "Distant metastasis__Presence": ["Suspicious", "Suspicious"],
    "Distant metastasis__Location": ["Liver", "Liver"],
    "Resectability": ["Resectable", "Resectable"],
})

result_df.columns = pd.MultiIndex.from_tuples(
    [tuple(col.split("__")) for col in result_df.columns]
)

result_df = result_df.transpose().fillna("").astype(str)

st.title("PDAC Report Summary")

# Streamlit에서 표시
st.dataframe(result_df, width=1200)
