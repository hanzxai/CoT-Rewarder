"""
好的，下面是对回答的分析和评判

```json
{
  "reason": "xxxxxxxxx",
  "consistency": true
}
```
还有其他问题吗？
提取string 提取json
"""


str1 = """```json 
{
  "reason": ["xxxxxxxxx"],
  "consistency": true
}
```"""
import re

def main():
    blocks = re.findall(r"'''(json)*'''",str1)
    rejson = []
    for  block in blocks:
        obj = json.loads(block)
        json.appen(obj)
    return rejson

