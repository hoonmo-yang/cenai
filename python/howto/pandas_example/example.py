all_keywords = [
    ["1", "2", "3",],
    ["4", "5", "6",],
]



keyword_texts = [
    "\n".join([
        f"<td>{keyword}</td>"
        for keyword in keywords
    ])
    for keywords in all_keywords
]

print(keyword_texts[0])
print(keyword_texts[1])