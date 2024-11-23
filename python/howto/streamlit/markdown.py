import streamlit as st

# HTML 콘텐츠
html_content = """
<h1>Sheet1</h1>

<h2>Tumor-vascular invasion</h2>

<h3>1) Artery</h3>
<p>SMA: {Abutment}</p>
<p>Celiac artery: {Abutment}</p>
<p>Common hepatic artery: {Abutment}</p>
<p>Proper hepatic artery: {Abutment}</p>
<p>1st Jejunal artery: {Abutment}</p>

<h3>2) Vein</h3>
<p>Poral vein: {No}</p>
<p>SMV: {Abutment}</p>
<p>1st Jejunal vein: {No}</p>
<p>IVC: {No}</p>
<p>Other veins: {No}</p>

<h3>Aorta</h3>
<p>{Abutment}</p>

<h3>Specify other veins</h3>
<p>{…}</p>
"""

# Streamlit 앱
st.set_page_config(layout="wide")  # 브라우저 화면을 절반 정도 크기로 설정
st.markdown(html_content, unsafe_allow_html=True)  # HTML 렌더링
