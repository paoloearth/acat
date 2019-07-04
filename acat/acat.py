# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure


#--------------------------------------------------------------------------------------------------------------------------------------
def DF(s, k):
    """
        Return a set in a list shape
        of distinct factors of length 
        k of the string s
        --------
        Parameters:
            s (str)
            k (int)
        
        eg:
        DF("BANANA", 2)
        Output:
        ['NA', 'BA', 'AN']
    """
    n=len(s)
    return list(set([s[i:i+k] for i in range(0, max(0, n-k+1))]))
#--------------------------------------------------------------------------------------------------------------------------------------
def DF_Cardinality(s):
    """
        Return the cardinality of the distinct factor 
        sets of the substrings of size 1 to len(s). 
        It is based on DF function.
        --------
        Parameters:
            s (str)
        
        eg.
        For "BANANA" we will have:
        1 ['B', 'A', 'N']
        2 ['BA', 'AN', 'NA']
        3 ['BAN', 'NAN', 'ANA']
        4 ['ANAN', 'NANA', 'BANA']
        5 ['BANAN', 'ANANA']
        6 ['BANANA']
        
        and so the output will be:
        DF_Cardinality("BANANA")
        Output:
        [3, 3, 3, 3, 2, 1]

    """
    n=len(s)
    l=list()
    for i in range(1, len(s)+1):
        l.append(len(DF(s, i)))
    return l
#--------------------------------------------------------------------------------------------------------------------------------------

def PV(s, alphabet) :
    """
        Return the parikh vector (list) of 
        the string s
        --------
        Parameters:
            s (str)
            alphabet (str)
            
        eg.
        PV("BANANA","ABN")
        Output:
        [3, 1, 2]
        
   """
    l=list()
    for c in alphabet:
        l.append(s.count(c))
    return l 
#--------------------------------------------------------------------------------------------------------------------------------------
def PV_atK(s, alphabet, k):
    """
        Calculate the Parikh Vector of all the
        substring of length k and return a set of it 
        -----------------------------------------
        Parameters:
            s (str)
            k (int) < len(s)
            alphabet (str)
            
            eg.
            PV_atK("BANANA", "ABN", 3)
            we have these subs ['BAN', 'NAN', 'ANA']
            
            output is a set and it is:
            {(1, 0, 2), (1, 1, 1), (2, 0, 1)}
    """
    dim=alphabet.find(max(s))+1
    r=list()
    for x in DF(s, k):
        r.append(PV(x, alphabet))
    return set(tuple(x) for x in r)
#--------------------------------------------------------------------------------------------------------------------------------------
def PV_Cardinality(s, ALPHABET): 
    """
       Return a list  containing the cardinality
       of the sets of PV of the substring of s
       from length 1 to length n-1
       
        -----------------------------------------
        Parameters:
            s (str)
            alphabet (str)
       
       eg.
       PV_Cardinality("BANANA","ABN")
       
       output is this list:
       [3, 2, 3, 2, 2, 1]   
            
      
    """
   
    
    lengths=list()
    for k in range (1, len(s)+1):
        lengths.append(len(PV_atK(s,ALPHABET,k)));
    return lengths

#--------------------------------------------------------------------------------------------------------------------------------------

def myplot(s, ALPHABET): #pk_sequence
    """
       Plot the  #k factors and the PARIKH VECTORS
       of the string s
        -----------------------------------------
        Parameters:
            s (str)
            alphabet (str)
            
        Output:
        plot with 2 lines 
    """
        
    if  set(s).issubset(set(ALPHABET))==False:
        print("The chars of the s are not contained entirely into the alphabet")
        return -1
    DF=DF_Cardinality(s)
    PV=PV_Cardinality(s, ALPHABET)
    df=pd.DataFrame({'x': range(1, max(len(DF), len(PV))+1), 'ParikhVectors':PV, 'DifferentFactors': DF})
    fig = plt.figure(num=None, figsize=(16, 5), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle("S: ["+s+ "] ROOT: ["+ getroot(s)+"]", fontsize=18)
    
    #--------------------------GRAPHIC OPTIONS AND SETTINGS--------------------------------
    # multiple line plot
    ax = fig.add_subplot(1, 1, 1)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, len(DF), 1)
    minor_ticks = np.arange(0, 101, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')
    ax.set_facecolor("white")
    plt.plot( 'x', 'ParikhVectors', data=df, marker='o', color='green', linewidth=2, fillstyle='full')
    plt.plot( 'x', 'DifferentFactors', data=df, marker='o', color='red', linewidth=2, fillstyle='full')
    plt.grid(color='salmon', linestyle='-.', linewidth=0.8,  which='both')
    # Set axis limits to show the markers completely
    plt.xlim(1, len(PV)+1)
    plt.ylim(1, max( max(PV) , max(DF) )+1)
    plt.legend()
#--------------------------------------------------------------------------------------------------------------------------------------
def DF_uptoK(s, k):
    """
       Return a list of lists which contains
       for each length i from 1 to k, the DFs
       of the string
        -----------------------------------------
        Parameters:
            s (str)
            k (int)
    """
        
    A=[]
    k=k+1
    for i in range(1, k):
        B=[]
        for element in DF(s, i):
            B.append(element)
        A.append(B)

    return A    
#--------------------------------------------------------------------------------------------------------------------------------------
def wordsgenerator_uptoK(alphabet, length):
    """
       Generate all the permutations of 
       the chars of the string length length 
       and output them into a list
       -----------------------------------------
        Parameters:
            alphabet (str)
            length (int)
        
        eg.
        wordsgenerator_uptoK("AB",2)
        
        output:
        ['AA', 'AB', 'BA', 'BB']
                 
            """
    c = [[]]
    for i in range(length):
        c = [ [x]+y     for x in alphabet      for y in c]
        
    wordsofsize=list()
    for w in c:
        wordsofsize.append(''.join(w))
    return wordsofsize
#--------------------------------------------------------------------------------------------------------------------------------------
def canbeextended(values, wordstocheck, alphabet):
    """Function that serves to build the extensibility
       table"""     
    r=[0] * len(values)
    for index, valore in enumerate(values):
        for simbolo in alphabet:
            if valore+simbolo in wordstocheck:
                r[index]+=1
        if r[index]>1:
            r[index]='+'
        elif r[index]==0:
            r[index]='o'
        else:
            r[index]='-'
    return r
#--------------------------------------------------------------------------------------------------------------------------------------
def extensibilitytable(S, alphabet):
    
    """
        Given a string and an alphabet, it builds the 
        extensibilitytable out of it
       -----------------------------------------
        S (str)
        alphabet (string)
        
        + if the word can be extended in more 
          than one way
        - if the word can be extended in just one
          way
        ° if the word cannot be extended
            
    """
    SIGMA=list(set(alphabet))
    A=DF_uptoK(S, len(S))

    D=[[]]
    for row in A:
        D.append(canbeextended(row, S, SIGMA))
    del D[0]
    k=1
    for i in range(0, len(D)):
        A.insert(k, D[i])
        k+=2
    df = pd.DataFrame(A)
    df = df.transpose()
    #df=df.style.highlight_null(null_color='black')
    return viewtable(df)
#----------------------------------------------------------------------------------------------------------------------------------------
def viewtable(df):
    """
    Just a satellite function for extensibilitytable
    """
    df=df.style.highlight_null(null_color='black')
    return df

#----------------------------------------------------------------------------------------------------------------------------------------
def wordsgenerator_fromAtoB(alphabet, a , b):
    """
   Generate all the permutations of 
   the chars of the string from length a
   to length b and output them into a list
   of lists
    --------------------------------------
    Parameters:
            alphabet (str)
            a (int)
            b (int) >= a
    """
    b=b+1
    L=[]
    for i in range(a, b):
        L.append((wordsgenerator_uptoK(alphabet, i)))
    flat_list = [item for sublist in L for item in sublist]
    return flat_list

#---------------------------------------------------------------------------------------------------------------------------------------  
def fibword(n, A0, A1):
    """ Given  the first 2 characters
        A0 and A1, it returns the fibonacci string of 
        n concatenations
        -----------------------------------------
        Parameters:
            A0 (str)
            A1 (str)
            n (int) """
    A=[A0,A1]
    for i in range(2,n):
        A.append(A[i-1]+A[i-2])
    return "".join(A)
#--------------------------------------------------------------------------------------------------------------------------------------- 
def powerword(s, n):
    """Given a strings, return a concatenation of
       the string to itself n times
     -----------------------------------------
        Parameters:
            s (str)
            n (int) """
    return s*n

#--------------------------------------------------------------------------------------------------------------------------------------- 
def getsuffixes(S):
    """ Satellite function of getroot"""
    suf=[]
    for i in range(1,len(S)):
        suf.append(S[i:len(S)])
    return suf
def getprefixes(S):
    """ Satellite function of getroot"""
    pre=[]
    for i in range(0,len(S)):
        pre.append(S[0:i])
    return pre

def border(S):
    """ Satellite function of getroot"""
    pre=getprefixes(S)
    suf=getsuffixes(S)
    A=set(pre)
    B=set(suf)
    L=list(A.intersection(B))
    if len(L)==0:
        print("There is no border");
        return -1
    l=len(max(L,key=len))
    w=max(L,key=len)
    
    return l, w

def getroot(S):
    """
      Given a string, it tries to find the root of it
      NB: works only for integer powers periodic strings
      -------------------------------------------------
      Parameters:
            S (str)
            
      eg.
      getroot("ababababababab")
      "ab"
     """
    if (border(S) == -1):
        return -1
    return S[0: (len(S)-border(S)[0])]
#------------------------------------------------------------------------------------------------------------------------------------
def FIRST(S, A, dim):
    
    """
    Function ad hoc to prove the conjecture
    that says that #PV=1 at every (m*k)
    ---------------------------------------
    Parameters:
    S (str)
    A (str)
    dim (integer)
    """
    print("The string is: [", S,"]")
    print("The length is: [", len(S),"]")
    print("The root is: [", getroot(S),"]")
    print("The period length is: [", len(getroot(S)),"]")
    print("For the values : [", len(getroot(S)),"] ... [", len(getroot(S))*2,"] ... [", len(getroot(S))*3,"] and so on we will always get just one type of PV as the different factors collapse" )
    L1=list()
    L2=list()
    indexes=list()
    t=DF(S,dim)
    for vettore in t:
        pv=PV(vettore, A)
        L1.append(pv)
        #L2.append(S.index(vettore))
        generator=find_all(S,vettore)
        temp=list()
        for i in generator:
            temp.append(i)
        indexes.append(temp)

    labels = ['S_Rotations', 'PV', 'index']
    df=pd.DataFrame(list(zip(t,L1, indexes)), columns=labels)
    print("Number of distinct PV: [" ,len(df.PV.apply(str).unique()), "]")
    #return df.sort_values('index')
    return df

def find_all(S, sub):
    """
    Satellite function of the function 
    """
    start = 0
    while True:
        start = S.find(sub, start)
        if start == -1:
            return -1
        yield start
        start += 1 # use start += 1 to find overlapping matches
        
        
def getalphabet(a):
    """ Given a string, returns the sorted alphabet
        in the form of a string """
    S="".join(sorted( list( set( list(a) ) ) ) )
    return S
#----------------------------------------------------------------------------------------------------------------------------------
def myplot_list(L, ALPHABET, power): 
    """
    Function that do the same of myplot but 
    for a list of words L of same length
    and their unique ALPHABET
    
    If power is one we keep only the root.
    -------------------------------------------------
    Parameters:
    L (list)
    ALPHABET (string)
    power (int)
            
    """
    T=[]
    roots=[]
    
    PWS=[]
    for s in L:
        PWS.append(powerword(s, power))
    L=[]
    L=PWS
    for word in PWS:
        T.append(PV_Cardinality(word, ALPHABET))
        roots.append(getroot(word))

    #--------------------------GRAPHIC OPTIONS AND SETTINGS--------------------------------
    fig = plt . figure ( figsize =(16 ,4))
    # multiple line plot
    ax = fig.add_subplot(1, 1, 1)
    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, len(max(L, key=len)), 1)
    minor_ticks = np.arange(0, 101, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')
    ax.set_facecolor("white")
  
    plt.grid(color='salmon', linestyle='-.', linewidth=0.8,  which='both')
    # Set axis limits to show the markers completely
    
    t2=[]
    for element in L:
        t2.append([0])
        
    xM2=np.array(t2);    
    xM=np.array(T);    
    
    xM=np.concatenate((xM2, xM), axis=1)

    fig.suptitle("DFs in Purple, PVs other colours",fontsize=14)
    
    
    M=0
    for i in range(0, len(xM)):
        plt.plot( xM[i] ,  linewidth =4, label=roots[i], alpha=0.8)
        if(M<max(DF_Cardinality(L[i]))):
            M=max(DF_Cardinality(L[i]))
        plt.plot( [0]+DF_Cardinality(L[i]) ,  linewidth =4, color="purple", alpha=0.7)    
       
   
    
    plt.xlim(1, len(L[0]))
    plt.ylim(1, M+1 )      
    plt.legend()
    plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------    
def TouchTheMax(L, ALPHABET,k):
    """
    Function used to investigate
    which words, at some point, has
    the #PVs equal to #DFs
    -------------------------------------------------
    Parameters:
    L (list)
            
    """    
    indexes=list()
    W=list()
    allW=list()

    for word in L:
        RED=DF_Cardinality(word)
        BLACK=PV_Cardinality(word, ALPHABET)
        M=max(RED)

        if M in BLACK:
            indexes.append([i for i,val in enumerate(BLACK) if val==M])
            W.append (str(word[0:k]))
        allW.append(str(word[0:k]) )  
    return W, indexes, allW

def set_approach(a,b):
    return list(set(a)-set(b))

def comparison(ALPHABET, k, power):
    """
    Function used to investigate
    which words, at some point, has
    the #PVs equal to #DFs
    -------------------------------------------------
    Parameters:
    ALPHABET (string)
    k (int) is the length of the string that has to be generated
    power (int) 
    """        
    PWS=[]
    L=wordsgenerator_fromAtoB(ALPHABET,k, k)
    L=sorted(L)
    for s in L:
        PWS.append(powerword(s, power))
    
    touch, indexes, allW=TouchTheMax(PWS, ALPHABET, k)
    nottouch=set_approach(allW,touch)
    
    print("Words where #PVs is equal to #Dfs at some point \n")
    print(*touch, sep=' - ')
    print("\n------------------------------------------------------\n")
    print("Words where #PVs is NOT equal to #Dfs at any point \n")
    print(*nottouch, sep=' - ')
    return touch, nottouch

#------------------------------------------------------------------------------------------------------------------
def df_from_alphabet(ALPHABET, k, power, bias):
    """
    Given an alphabet in string form
    and a length k, it computes all the
    words of length k, calculate their 
    PVs and return two lists. The first 
    is the list of the words. The second
    list is the #PVs at n-k+bias
    
    """
    PWS=[]
    L=wordsgenerator_fromAtoB(ALPHABET,k, k)
    roots=L
    L=sorted(L)
    for s in L:
        PWS.append(powerword(s, power))
    values=list()
    for word in PWS:
        ww=list()
        n=len(word)
        n_k=n-k+bias
        ww=PV_Cardinality(word, ALPHABET)
        values.append(ww[n_k]) 
    
    print("\n****************************************************\n")
    print("The size of the alphabet is [" +str( len(ALPHABET))+"]")
    print("The lengths of the strings is [" +str( n)+"]")
    print("The lengths of the roots of the strings is [" +str(k)+"]")
    print("We investigate position [" +str( n_k+1)+"]")
    print("\n****************************************************\n")
    return roots,values

def pd_grouping(A, B):
    labels=["word", "#PVs"]
    df = pd.DataFrame( {'word':A, 'PVs':B})
    s=df.groupby('PVs')['word'].apply(list)
    return s.tolist(), s

def investigate(ALPHABET,rootsize, power, bias=0):
    """
    Given an alphabet in string form
    and a length rootsize, it computes all the
    words of length k, calculate their 
    PVs and return two lists. The first 
    is the list of the words. The second
    list is the #PVs at n-k+bias
    -------------------------------------------------
    Parameters:
    ALPHABET (string)
    rootsize (int) is the length of the string that has to be generated
    power (int) 
    bias (int)
    """
    bias=bias-1
    A, B=df_from_alphabet(ALPHABET,rootsize, power, bias)
    L, df=pd_grouping(A, B) 
    
    for i in range(0, len(L)):
        print("Words with [", i+1, "] PVs")
        print(*L[i], sep=' - ')
        print("\n------------------------------------------------------\n")

#-----------------------------------------------------------------------------------

def rotations(S):
    """
    Returns all the rotations of a string
    """
    L=list(S)
    L2=list()
    for i in range(0, len(L)):
        L2.append(''.join(L[i:] + L[:i]))
    return L2