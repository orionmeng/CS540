import sys
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    with open (filename,encoding='utf-8') as f:
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            X[char] = 0
        for line in f:
            for char in line.upper():
                if char in X:
                    X[char] += 1

    return X

def print_character_count(dictionary):
    for char, count in dictionary.items():
        print(f"{char} {count}")

def main():
    filename = "./letter.txt"
    x = shred(filename)
    e, s = get_parameter_vectors()

    x_ln_e = 0
    x_ln_s = 0
    i_loop = 0
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        x_ln_e += x[char] * math.log(e[i_loop])
        x_ln_s += x[char] * math.log(s[i_loop])
        i_loop += 1
    F_of_e = math.log(0.6) + x_ln_e # P(Y = english) = 0.6
    F_of_s = math.log(0.4) + x_ln_s # P(Y = spanish) = 0.4

    e_given_x = 0
    if math.isinf(float(F_of_e)):
        if F_of_e > 0:
            e_given_x = 0
        else:
            e_given_x = 1
    elif F_of_s - F_of_e >= 100:
        e_given_x = 0
    elif F_of_s - F_of_e <= -100:
        e_given_x = 1
    else:
        e_given_x = 1 / (1 + math.exp(F_of_s - F_of_e))

    print("Q1")
    print_character_count(x)

    print("Q2")
    print("{:.4f}".format(x['A'] * math.log(e[0])))
    print("{:.4f}".format(x['A'] * math.log(s[0])))

    print("Q3")
    print("{:.4f}".format(F_of_e))
    print("{:.4f}".format(F_of_s))

    print("Q4")
    print("{:.4f}".format(e_given_x))

if __name__ == "__main__":
    main()
