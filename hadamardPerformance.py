# For each realization phi_1, ..., phi_K of random phases the Hadamard receiver gets coherent state inputs.
# Thus it produces coherent state outputs.
# Given any receiver, the probability of deciding for coherent state b given coherent state a is p_R(b|a).
# Given randomly chosen phases phi_1, ..., phi_K the received state $\tilde a(phi_1, ..., phi_K)$ is
#    a*2^{log(K)}*(sum[exp(phi_1),..., exp(phi_{K/2})] - sum[exp(phi_{K/2}),..., exp(phi_{K})]) (if one listens at the "wrong" output port)
#    a*2^{log(K)}*(sum[exp(phi_1),..., exp(phi_{K/2})] + sum[exp(phi_{K/2}),..., exp(phi_{K})]) (if one listens at the "correct" output port)
# Thus it is possible to calculate p_R(b|\tilde a)
# The probability of deciding for b is then
#    sum_{phi_1,...,phi_K} p_R(b|\tilde a(phi_1, ..., phi_K))


import scipy.special as scps
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.special import i1
import numpy as np
import random
import itertools
import collections
import math


def rX(a, K, kappa, equalPort=True):
    # calculates the received signal when listening at the right or at the wrong port under iid van Mises noise with kappa=kappa
    # K has to be a power of two!!
    rx = 0
    s = [0 for i in range(K)]
    if kappa != np.inf:
        s = np.random.vonmises(0, kappa, K)
        #print("K=",K,"random von mises numbers are",s)
    if equalPort:
        rx = a * sum([np.exp(-complex(0, s[i])) for i in range(K)])/np.sqrt(K)
        #print("TRUE: rX=",rx)
    else:
        diffSum = sum([np.power(-1,i)*np.exp(complex(0, s[i])) for i in range(int(K))])
        rx = a * diffSum/np.sqrt(K)
        #print("FALSE: difference sum is",diffSum)
        #print("FALSE: rx should be [",a*diffSum.real/np.sqrt(K),a*diffSum.imag/np.sqrt(K),"]" )
        #print("FALSE: a=",a)
        #print("FALSE: rX=",rx)
    return rx



def homodyne(a, displacement, epsilon):
    pMinusA = 0.5*( 1 - scps.erf(np.sqrt(2)*(a+epsilon)))
    pZero = 0.5*( scps.erf(np.sqrt(2)*(epsilon - a)) + scps.erf(np.sqrt(2)*(a+epsilon)))
    pPlusA = 0.5*( 1 - scps.erf(np.sqrt(2)*(epsilon - a)))
    #print("a=",a,"displacement=",displacement, "epsilon=",epsilon)
    #print("pMinus=",pMinusA.real,"pZero=",pZero.real,"pLusA=",pPlusA.real)
    coin = random.random()
    #print("coin=",coin)
    out = displacement
    if coin < pMinusA.real:
        out = -displacement
    elif coin >= pMinusA.real and coin < pMinusA.real + pZero.real:
        out = 0
    elif coin >= pMinusA.real + pZero.real:
        out = displacement
    #print("-:[0,",pMinusA.real,"], 0:[",pMinusA.real,pMinusA.real + pZero.real,"]","1:[",pMinusA.real + pZero.real,"1]")
    #print(out)
    return out

def sampledQ(e, K, kappa, samples):
    # samples the classical channel defined from using a Hadamard receiver with words of length K
    # the output has form [q( |a,k=l), q( |b,k=l), q( |a,k#l), q( |b,k#l)]
    # it holds a=sqrt(e) and b=-sqrt(e) (BPSK)
    # "k = l" indicates the probability distribution at the same port, k#l quantifies the statistics if one listens at the wrong port
    print("sampledQ thinks samples equals",samples)
    aa = 0
    ba = 0
    zeroA = 0
    ab = 0
    bb = 0
    zeroB = 0
    a0a = 0
    b0a = 0
    zero0a = 0
    a0b = 0
    b0b = 0
    zero0b = 0
    displ = e*np.sqrt(K)
    epsilon = displ/2
    for i in range(samples):
        # calculate what happens if "+" is sent
        a = displ
        b = -a
        # add noise to a
        rx = rX(a, K, kappa, True)
        #print(rx)
        # get output of homodyne receiver if listening at the CORRECT port & check if decoded correctly
        out = homodyne(rx, displ, epsilon)
        #print(out, a)
        if out == a:
            aa += 1
        elif out == b:
            ba += 1
        elif out == 0:
            zeroA += 1
        else:
            print(out)
            raise ValueError('Homodyne receiver gave strange output')
        # calculate what happens if "-" is sent
        # add noise
        rx = rX(b, K, kappa, True)
        #print("RIGHT: rx=",rx)
        out = homodyne(rx, displ, epsilon)
        # print(out, b)
        if out == b:
            bb += 1
        elif out == a:
            ab += 1
        elif out == 0:
            zeroB += 1
        else:
            print(out)
            raise ValueError('Homodyne receiver gave strange output')
        # get output of Homodyne receiver if listening at the WRONG port
        rx = rX(a, K, kappa, False)
        #print("WRONG: rx=",rx)
        out = homodyne(rx, displ, epsilon)
        #print("ratio=",rx.real/rX(a, K, kappa, True).real, "  out=",out, "rx=",rx.real,"epsilon=",epsilon)
        if out == a:
            a0a += 1
        elif out == b:
            b0a += 1
        elif out == 0:
            zero0a += 1
        rx = rX(b, K, kappa, False)
        # get output of homodyne receiver if listening at the WRONG port
        out = homodyne(rx, displ, epsilon)
        if out == a:
            a0b += 1
        elif out == b:
            b0b += 1
        elif out == 0:
            zero0b += 1
        q = [[[aa / samples, ba / samples, zeroA / samples], [ab / samples, bb / samples, zeroB / samples]],
             [[a0a / samples, b0a / samples, zero0a / samples], [a0b / samples, b0b / samples, zero0b / samples]]]
    return q


def condDistrib(a, bK, k, q):
    # the transmitted symbol at port k is a
    # here we are on a logical level, so a can be 0 or 1. Structure is as listed in the definition of q:
    # q = [[[aa / samples, ba / samples, 0a / samples], [ab / samples, bb / samples, 0b / samples]], 
    #       [[a0a / samples, b0a / samples, 00a / samples], [a0b / samples, b0b / samples, 00b / samples]]]
    # the distribution at the correct output port is different from that at all other ports, thus the symbol at that port gets special treatment
    resultAtCorrectPort = int(bK[k])
    count = collections.Counter(bK)
    #print("checking count:",count,sum(count))
    nPlus1 = count[0]
    #print(nPlus1)
    nMinus1 = count[1]
    #print(nMinus1)
    nZero = len(bK) - nPlus1 - nMinus1

    if resultAtCorrectPort == 0:
        nPlus1 -= 1
    if resultAtCorrectPort == 1:
        nMinus1 -= 1
    if resultAtCorrectPort == 2:
        nZero -= 1
    #print(nPlus1 + nMinus1 + nZero)
    # probability of detecting b given a at the RIGHT port is q[0][a][b]
    # probability of detecting b given a at the WRONG port is q[1][a][b]
    plus = [q[1][a][0] for i in range(nPlus1)]
    minus = [q[1][a][1] for i in range(nMinus1)]
    zero = [q[1][a][2] for i in range(nZero)]
    #print(plus, minus, zero)
    prodPlus = 1
    prodMinus = 1
    prodZero = 1
    if len(plus) > 0:
        prodPlus = q[1][a][0]**nPlus1#np.prod(plus)
    if len(minus) >0:
        prodMinus = q[1][a][1]**nMinus1#np.prod(minus)
    if len(zero) > 0:
        prodZero = q[1][a][2]**nZero#np.prod(zero)
    out = q[0][a][resultAtCorrectPort]*prodPlus*prodMinus*prodZero
    #print(bK,out)
    return out


def outDistrib(q, bK):
    # here we are on a logical level, so a can be 0 or 1
    K = len(bK)
    oD = (1 / K) * (1 / 2) * sum([condDistrib(a, bK, k, q) for k in range(K) for a in range(2)])
    return oD

def pLogP(p):
    out = 0
    if p > 0:
        out = p * np.log2(p)
    return out

def size(measType):
    # alphabet is {0,1,2}^K
    length = sum([x for x in measType])
    nPlus1  = measType[0]
    nMinus1 = measType[1]
    return scps.binom(length, nPlus1)*scps.binom(length - nPlus1, nMinus1)

def mutualInformation(q, K):
    condH = 0
    outH = 0
    if K > 1:
        # we look at the cases where the correct port is - without loss of generality - port 0
        # define how many 1's are detected (logical 0), at ports other than port 0:
        for i in range(K):
            # define how many -1's are detected (logical 1), at ports other than port 1:
            for j in range(K - i ):
                # define what was sent at port 0:
                for a in range(2):
                    # define what was received at port 0:
                    for b in range(3):
                        # create string of measurements, without loss of generality well-ordered:
                        resultAtPortK = [b]
                        bK = resultAtPortK + [ 0 for m in range(i)]
                        bK = bK + [ 1 for m in range(j)         ]
                        # define how many 0's (logical 2) are detected:
                        bK = bK + [ 2 for m in range(K - 1 - i - j)  ]
                        # assume the correct port (port 0) was chosen by the sender:
                        #print("K=",K,"bK=",bK)
                        cd = condDistrib(a, bK, 0, q)
                        if cd > 0:
                            #print("K=",K,"i=",i,"j=",j,[ i, j, K - 1 - i - j ],"has size",size([ i, j, K - 1 - i - j ]),"bK=",bK)
                            #print("cd=",cd)
                            condH -= size([ i, j, K - 1 - i - j ]) * cd * np.log2(cd)
    # there is a total of 2 phases available for the receiver, so we have to divide by two:
    condH = condH/2
    # there are a total of 8 ports, each of them behaves identically, and the sender averages uniformly over the ports. Thus condH is calculated now.
    # the output entropy:
    if K > 1:
        st1 = 0
        stAS = 0
        # we look at the cases where the correct port is - without loss of generality - port 0
        # define how many 1's are detected (logical 0), at ports other than port 0:
        for i in range(K):
            # define how many -1's are detected (logical 1), at ports other than port 1:
            for j in range(K - i ):
                bK = [ 0 for m in range(i)]
                bK = bK + [ 1 for m in range(j) ]
                # define how many 0's (logical 2) are detected:
                bK = bK + [ 2 for m in range(K -1 - i - j)  ]
                # now average over the input variables:
                #print("calculating output entropy for bK=",bK,"i=",i,"j=",j)
                for b in range(3):
                    # now bK is defined up to permutation on "zero output ports"
                    od = 0
                    for a in range(2):
                        for k in range(K):
                            cd = condDistrib(a, [b] + bK, k, q)
                            if cd > 0:
                                od += cd
                    od = od/(K*2)
                    st1 += size([i, j, K - 1 - i - j]) * od
                    #stAS += size([ i, j, K - i - j ])
                    if od > 0:
                        #print( - size([ i, j, K - i - j ]) * od * np.log2(od))
                        outH -= size([i, j, K - 1 - i - j])* od * np.log2(od)
    print("sum-to-one test:",st1)
    print("sum to alphabet size test:",stAS)
    print("outH=",outH)
    print("condH=",condH)
    return outH - condH

def capacity(q, K):
    return mutualInformation( q, K)/K

def shannonCapacity( kappa, e, samples=1000 ):
    s = samples
    qfull = sampledQ( e, 1, kappa, s )
    q = qfull[0]
    #print(qfull)
    cap = 0
    for i in range(s):
        p = i/s
        outD = [p*q[0][0] + ( 1 - p )*q[1][0], p*q[0][1] + ( 1 - p )*q[1][1]]
        outH = - pLogP(outD[0]) - pLogP(outD[1])
        condH = - p*( pLogP(q[0][0]) + pLogP(q[0][1]) ) - (1-p)*( pLogP(q[1][0]) + pLogP(q[1][1]) )
        if outH - condH > cap:
            cap = outH - condH
    return cap

def kappa(N, B):
    # N is number of photons per second, B is the baud rate
    # If both N and B are given as 10**x where x is an integer then things work as expected
    n = math.log(N, 10)
    b = math.log(B, 10)
    return np.power(10, 41 - n - 2*b)
    #N is the number of photons per second and B the baud rate
    
def g (x):
    return np.log2( 1 + x) + x*np.log2(1 + 1/x)

# this is the spectral efficiency according to the Holevo formula
def sHol (tau, ns, noise):
    if noise>0:
        return g( ns*tau + noise ) - g( noise)
    else:
        return g( ns*tau + noise )

if __name__ == "__main__" :

        
    baudrateCalculations = False
    if baudrateCalculations:
        kap = 0.001
        A = np.exp(-0.05*220)*10**16
        print("lowest expected photon numer at receiver is",A/((100+20*10)*(10**9)))
        samples = 1000
        print("A=",A,"kappa=",kap,"samples=",samples)
        holData = []
        shData = []
        for b in range(20):
            br = (100+b*10)*(10**9)
            kap = kappa(10**16,br)
            K = int(np.power(2,2))
            holAvg = 0
            shAvg = 0
            holCapacities = []
            shCapacities = []
            for step in range(20):
                q = sampledQ( A/br ,K ,kap ,samples)
                m = mutualInformation(q, K)
                holCap = m/K
                holAvg += holCap
                holCapacities += [holCap]
                shCap = shannonCapacity(kap,A/br)
                shAvg += shCap
                shCapacities += [shCap]
                #print("at baudrate=",br,"we have photon number ",A/br)
                #print("at baudrate=",br,"we have Hadamard capacity",br*m/K)
                #print("at baudrate=",br,"we have Shannon capacity",br*shannonCapacity(kap,A/br))
            holVariance = 0
            holAvg = holAvg/(20)
            shVariance = 0
            shAvg = shAvg/20
            for step in range(20):
                holVariance += (holAvg - holCapacities[step])**2
                shVariance += (shAvg - shCapacities[step])**2
            shVariance = np.sqrt(shVariance/(20-1))
            holVariance = np.sqrt(holVariance/(20-1))
            print("at baudrate=",br,"Shannon average=",shAvg,"variance=",shVariance)
            print("at baudrate=",br,"Holevo average=",holAvg,"variance=",holVariance)
            holData += [[br, br*holAvg, br*holVariance]]
            shData += [[br, br*shAvg, br*shVariance]]
        print(holData)
        print(shData)



    tau = np.exp(-0.05*600)
    power = (10**22)
    A = tau*power
    samples = 1000
    br = (4000)*(10**9)
    kap = kappa(10**16,br)
    K = int(np.power(2,3))
    #print(rX(A/br, K, kap, equalPort=True))
    #print(rX(A/br, K, kap, equalPort=False))
    #print("getting q")
    q = sampledQ( A/br ,K ,kap ,samples)
    condDistribTest = 0
    for i in range(K ):
        # define how many -1's are detected (logical 1), at ports other than port 1:
        for j in range( K - i ):
            bK = [ 0 for m in range(i)]
            bK = bK + [ 1 for m in range(j) ]
            # define how many 0's (logical 2) are detected:
            bK = bK + [ 2 for m in range(K -1 - i - j)  ]
            # now average over the input variables:
            od = 0
            #print("calculating output entropy for bK=",bK,"i=",i,"j=",j)
            for b in range(3):
                n0 = i
                n1 = j
                condDistribTest += size([n0, n1, K - 1 - n0 - n1])*condDistrib(1, [b] + bK, 0, q)
    print("cd test",condDistribTest)

  
    #print(q)
    m = mutualInformation(q, K)
    print(m)
    print(homodyne(0, 1, 0.5))
    print(sHol(tau,power/br,0))
    print(" kappa=",kap,
          "\n Hadamard order is",K,
          "\n Hadamard Capacity is",math.log(br*m/K, 10),
          "\n Hadamard efficiency is",m/K,
          "\n photon number at receiver is ",A/br,
          "\n Shannon capacity is",math.log(br*shannonCapacity(kap,A/br,samples), 10))

