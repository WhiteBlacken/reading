def isValid(s: str) -> bool:
    tmp = []
    dict = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    for i in range(len(s)):
        print(tmp)
        if s[i] in ['(', '[', '{']:
            tmp.append(s[i])
        elif inlist(dict[s[i]], tmp):
            print("remove")
            tmp.remove(dict[s[i]])
        else:
            return False
    return not tmp


def inlist(num, list):
    return any(num == item for item in list)


s="()"
print(isValid(s))