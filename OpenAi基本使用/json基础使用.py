import json
d = {
    "name" : "4-",
    "age" : 18,
    "gender" : "男"
}
s = json.dumps(d, ensure_ascii=False)
print(s)