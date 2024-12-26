import io
import pdfkit
from pathlib import Path
from PyPDF2 import PdfMerger


html = '''
<!DOCTYPE html>
<html lang="kr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> 연구결과 요약문 </title>

        <style>
body {
    margin: 5px;
    font-family: 'Noto Sans KR', sans-serif;
}
h1, h2 {
    font-size: 18px;
    color: #333;
}
.section1 {
    font-size: 90%;
    margin-bottom: 5px;
    width: 90%;
}
.section2 {
    font-size: 90%;
    margin-bottom: 5px;
    width: 90%;
}

.key-value {
    display: flex;
    justify-content: space-between;
    margin-left: 5px;
}
.key-value span {
    flex: 1;
}

table {
border-collapse: collapse;
width: 100%;
margin: 10px 0;
box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
table-layout: auto;
max-width: 100%;
}

th, td {
    border: 1px solid #ddd;
    padding: 4px 5px;
    text-align: left;
    vertical-align: top;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

th {
    background-color: #f4f4f4;
    padding: 4px 5px;
    color: #333;
    font-weight: bold;
    vertical-align: top;
}

.section1 td:nth-child(1) {
    width: 15%;
}

.section2 td:nth-child(1) {
    width: 15%;
}

.section2 td:nth-child(2) {
    width: 5%;
}

.section3 td:nth-child(1) {
    width: 15%;
}

tr:nth-child(even) {
    background-color: #f9f9f9;
}

tr:hover {
    background-color: #f1f1f1;
}

td {
    transition: all 0.3s;
}
</style>

        <link rel="font-kr" href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
    </head>
    <body>
        <h2>연구과제 개요</h2>
        <div class="section1">
            <table>
            <tr>
                <td rowspan="2">연구개발과제명</td>
                <td colspan="4">[국문]&nbsp;&nbsp;비교-기능 유전체 분석을 통한 식물의 비기주저항성 유전자 동정 및 작용기작 연구</td>
            </tr>
            <tr>
                <td colspan="4">[영문]&nbsp;&nbsp;Comparative functional genomics for identifying plant nonhost resistance genes and molecular mechanism</td>
            </tr>
            <tr>
                <td>주관연구개발기관</td>
                <td colspan="4">서울대학교</td>
            </tr>
            <tr>
                <td rowspan="2">연구책임자</td>
                <td>성명</td>
                <td>최도일</td>
                <td>직급(직위)</td>
                <td>교수</td>
            </tr>
            <tr>
                <td>소속부서</td>
                <td>농업생명과학대학</td>
                <td>전공</td>
                <td>기타농생물</td>
            </tr>
            </table>
        </div>

        <h2>연구결과 요약문 유사도</h2>
        <div class="section2">
            <table>
            <tr>
                <td>File</td>
                <td colspan="2">sample/IR_00000000012159757_20240320093725_965639.hwp</td>
            </tr>
            <tr>
                <td colspan="3">유사도 평가 (Score: 0~10)</td>
            </tr>
            <tr>
                <td>항목</td>
                <td>점수</td>
                <td>차이점</td>
            </tr>
            <tr>
                <td>연구개요</td>
                <td>7.0</td>
                <td>두 요약 모두 NLR-omics와 비기주저항성 인자 동정에 대한 내용을 포함하고 있지만, AI 작성 요약은 연구의 구체적인 방법과 가설 제시에 더 초점을 맞추고 있으며, 직접 작성 요약은 시스템적 해석 및 농업적 활용성에 더 중점을 두고 설명하고 있습니다.</td>
            </tr>
            <tr>
                <td>연구 목표대비 연구결과</td>
                <td>8.0</td>
                <td>두 요약 모두 주요 연구 목표와 성과를 설명하고 있지만, AI 작성 요약은 NLR-network와 구조적 및 기능적 상동성에 대한 추가적인 언급이 있으며, 직접 작성 요약은 연구 목표의 달성 정도에 대한 구체적인 수치와 논문 출판 성과에 더 강조를 두고 있습니다.</td>
            </tr>
            <tr>
                <td>연구개발성과의 활용 계획 및 기대효과(연구개발결과의 중요성)</td>
                <td>7.0</td>
                <td>두 요약 모두 연구개발성과의 중요성을 식량위기 해결과 저항성 작물 개발에 두고 있지만, AI 작성 요약은 구체적인 저널명과 특허 출원 계획을 언급하며 실용적 활용 계획에 더 자세히 기술하고 있습니다.</td>
            </tr>
            <tr>
                <td>중심어</td>
                <td>6.0</td>
                <td>두 요약 모두 비기주저항성, NLR 및 병원균 관련 키워드를 포함하고 있지만, 직접 작성 요약은 병원성단백질과 가지과식물 등 좀 더 일반적인 범주의 용어를 사용한 반면, AI 작성 요약은 형질전환 식물체와 effector 동정 등 좀 더 구체적인 연구 관련 용어를 사용하고 있습니다.</td>
            </tr>
            </table>
        </div>

        <h2>연구결과 요약문</h2>
        <div class="section3">
            <table>
            <tr>
                <td></td>
                <td>직접 작성</td>
                <td>AI 작성</td>
            </tr>
            <tr>
                <td>연구개요</td>
                <td>Omics적 접근을 통해 식물의 면역수용체인 NLR과 병원균의 병원성단백질인 effector를 재료로 비기주저항성이 매개되는데 있어서 식물의 저항성 유전자와 병원균의 저해기작이 기여하는 바에 대한 시스템적인 해석 및 실용(농업)적 측면에서의 활용성 제시</td>
                <td>이 연구는 NLR-omics와 비교유전체 연구를 통해 다양한 식물의 비기주저항성인자를 동정하고 작용기작을 구명하는 것을 목표로 한다. 특히, 감자역병균과 고추를 모델로 하여 비기주저항성 유전자를 탐색하고, 이를 기반으로 저항성 식물체를 제작하여 실질적인 작물보호 수단으로 활용하고자 한다. 연구는 NLR과 effector 간의 상호작용을 분석하고, 유전자 기능을 검증하여 비기주저항성의 이론적 기반을 확립하는 데 주력한다. 연구진은 고추의 NLR이 감자역병균에 대한 저항성을 매개할 수 있음을 확인하고, 이를 통해 NLR-effector의 종간 인식 작용이 비기주저항성의 한 축을 이룬다는 가설을 제시하였다. 연구는 모든 목표를 100% 이상 달성하였으며, 관련 연구 결과는 다수의 논문으로 출판되었다.</td>
            </tr>
            <tr>
                <td>연구 목표대비 연구결과</td>
                <td>전체 연구기간(2021~2024) 동안의 주요목표로 제시되었던 세 가지 항목: 1. NLR-omics 및 비교유전체 연구를 통한 다양한 식물의 비기주저항성인자 동정 2. 형질전환을 통한 비기주저항성의 작물보호 수단으로서의 기능증명 및 기술확보 3. 비기주저항성 관련 수월성이 있는 논문 출판 (IF > 7, 2편 이상) 들에 대해서 ‘감자역병균에 대해 기능하는 고추의 저항성 유전자를 다수 동정’하였고 ‘발굴된 유전자들을 발현하는 감자 및 담배 형질전환체에서 저항성을 확인’하였으며 ‘IF > 7 이상의 논문 6편을 포함하여 총 8편의 논문을 주요저자로서 출판’ 함으로써 연구과제의 커다란 목표를 모두 초과달성 하였다고 판단됨. 이외에도 연구 계획서에 제시되었던 세부목표들 또한 해당 내용들과 관련이 깊은 연구 수행을 통해 논문화되거나 유의미한 결론을 도출해낼 수 있었음. 최종적으로, 본 연구진은 지난 3년 동안의 중견 연구과제를 통해 비기주저항성에 식물의 면역 수용체와 병원균의 이펙터가 관여하는 원리를 구명해내었으며, 이를 농업적으로 활용하기 위한 이론적인 기반을 제시하였다고 판단됨.</td>
                <td>이 연구는 감자역병균에 대한 고추와 기타 가지과 식물의 비기주저항성 유전자를 동정하고, 이들의 작용 기작을 규명하는 것을 목표로 한다. 연구 결과, 고추의 NLR-network를 추가 연구하여 감자역병균에 대한 저항성을 나타내는 유전자를 동정하였으며, 이는 야생 감자 NLR들과 구조적 및 기능적 상동성을 가지는 것으로 확인되었다. 또한, Avrblb2와 같은 effector가 식물 면역 반응을 억제하는 메커니즘을 분자 수준에서 규명하였다. 다양한 Phytophthora 병원균에서 비기주저항성을 나타내는 effector를 식별하고, 이들이 가지과 식물의 NLR 유전자와 상호작용하여 면역 반응을 유도하는지 확인하였다. 연구는 비기주저항성 유전자가 기주저항성 유전자와 어떻게 상호작용하는지를 분석하고, 이를 통해 비기주저항성의 분자적 기작을 규명하는 데 기여하였다.</td>
            </tr>
            <tr>
                <td>연구개발성과의 활용 계획 및 기대효과(연구개발결과의 중요성)</td>
                <td>본 연구과제의 가장 주요한 성과 중 하나인 감자역병균에 대한 고추의 저항성 유전자들을 최종적으로 감자 및 토마토와 같은 농업/경제적으로 중요한 위치의 작물들에 형질전환 시킴으로서 전세계적으로 매년 발생되는 약 200억 달러(90B 달러의 약 15%) 가량의 Phytophthora에 의한 피해를 줄이는데 기여할 수 있을 것으로 생각됨. 또한 본 연구결과는 감자역병균에 의해 발생되는 피해 뿐만이 아니라 다양한 병원균에 대항하는 지속가능한 병저항성 유전자의 탐색을 기존의 방식보다 더 넓은 종의 식물로부터 가능케하여 최종적으로는 다양한 병원균에 의해 발생되는 작물 생산량의 피해 감소에 기여함으로서 전세계적으로 대두되는 문제인 식량 수요량의 증대를 해결하는데 있어 주요한 해결책이 될 수 있을 것으로 기대됨.</td>
                <td>연구개발성과는 식량위기 해결에 기여할 수 있는 중요성을 지니고 있다. 특히, 비기주 식물에서 NLR을 동정하여 감자역병균 저항성을 매개하는 기술을 개발함으로써 다양한 병원균에 대한 저항성 작물의 개발이 가능해질 것으로 기대된다. 연구 결과는 Plant Biotechnology Journal, New Phytologist, Plant Communication 등의 저널에 게재되어 연구의 수월성과 중요성을 입증하였다. 연구성과의 관리 및 활용 계획으로는 고추에서 동정된 감자역병균 저항성 유전자와 다양한 Phytophthora에 대해 기능하는 유전자를 통해 형질전환 식물체를 제작하고, 이들을 검증하는 단계에 도달하였다. 최종적으로 특허를 출원할 계획이며, 이러한 기술은 병저항성 작물 육종 프로그램에 활용될 수 있을 것이다.</td>
            </tr>
            <tr>
                <td>중심어 (국문)</td>
                <td style="text-align: justify;">비기주저항성, 면역수용체, 병원성단백질, 가지과식물, 병원성난균류</td>
                <td style="text-align: justify;">비기주저항성, NLR 네트워크, 감자역병균, 형질전환 식물체, Effector 동정</td>
            </tr>
            <tr>
                <td>중심어 (영문)</td>
                <td style="text-align: justify;">Nonhost resistance, NLR, Effector, Solanaceae, Phytophthora</td>
                <td style="text-align: justify;">Nonhost resistance, NLR network, Phytophthora infestans, Transgenic plants, Effector identification</td>
            </tr>
        </div>
    </body>
</html>
'''

pdf_buffer = io.BytesIO(pdfkit.from_string(html, False))

merger = PdfMerger()
merger.append(pdf_buffer)

pdf = Path("output.pdf")

with pdf.open("wb") as fout:
    merger.write(fout)

print("success")