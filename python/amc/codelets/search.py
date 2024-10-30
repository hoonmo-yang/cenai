from rapidfuzz import fuzz, process

sentence = '''
1 연구개발과제의개요             
나는 민족 중흥의 역사적 사명을 띄고 쏼라 쏼라
연구 과제 개요는 이렇다

2. 연구개발 성과의 관련 분야에 대한 기여 정도   aaa  


5. 연구개발 성과의 관리 및 활용 계획 (3)
5.1 aaa
5.2 kkk
5.3
5.4
'''

keywords = [
    "연구개발 과제의 개요",
    "연구개발 성과의 관리 및 활용 계획",
    "참조문헌",
]

results = process.extract(sentence, keywords, scorer=fuzz.partial_ratio)

for result in results:
    keyword, score, index = result
    print(f"keyword:{keyword} socre:{score} original:{keywords[index].strip()}")
