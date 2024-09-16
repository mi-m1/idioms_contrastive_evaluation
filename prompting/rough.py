import re
def check_model_output(output):

    if ' i' in output or 'idiomatic' in output or 'figurative' in output or 'figuratively' in output:
        pred = 'i'
        return pred

    elif 'l ' in output or 'literal' in output or 'literally' in output:
        pred = 'l'
        return pred

    else:
        print("wajt")

    

string = "Just generate the letter without explanations. ['i']"

match = re.search(r'\[\s*[\'\"](i|l)[\'\"]\s*\]', string)
if match:
    result = match.group(1)
    print(f"this is result")
    # print(len(result))
    print(result)
    # print(check_model_output(result))

else:
    print("No match found")


# A:\s*(.*)
# captures:"l (The expression is used literally in this sentence)" 

# Just generate the letter object without explanations: A: l (The expression is used literally in this sentence)
# Just generate the letter object without explanations: A:l (The expression is used literally in this sentence)


