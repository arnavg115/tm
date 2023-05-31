import re


with open("requirements.txt", "r") as fil:
    x = re.sub("==.*", "", fil.read())

# fil.write()
with open("requirements.txt", "w") as fil:
    fil.write(x)
