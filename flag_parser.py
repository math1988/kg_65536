import sys

# Takes in an array of arguments, and return a dictionary of key-value pairs.
# For example ["--input=a","--output=b"] will result in
# {"input":"a", "output":"b"}
def parse_flag(arg_list):
    result = {}
    for arg in arg_list:
        if arg[0:2] != "--" :
            continue
        equal_position = arg.find("=")
        if (equal_position==-1):
            continue
        key = arg[2 : equal_position]
        value = arg[equal_position+1 : ]
        result[key]=value
    return result

# Main function is test purpose only
if __name__ == "__main__":
    print(parse_flag(sys.argv))