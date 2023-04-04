#The following license applies:
##MIT License
##Copyright (c) [2022] [Zuhra Amiri and Janis Nötzel]
##Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
##The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
##THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# For each realization phi_1, ..., phi_K of random phases the Hadamard receiver gets coherent state inputs.
# Thus it produces coherent state outputs.
# Given any receiver, the probability of deciding for coherent state b given coherent state a is p_R(b|a).
# Given randomly chosen phases phi_1, ..., phi_K the received state $\tilde a(phi_1, ..., phi_K)$ is
#    a*2^{log(K)}*(sum[exp(phi_1),..., exp(phi_{K/2})] - sum[exp(phi_{K/2}),..., exp(phi_{K})]) (if one listens at the "wrong" output port)
#    a*2^{log(K)}*(sum[exp(phi_1),..., exp(phi_{K/2})] + sum[exp(phi_{K/2}),..., exp(phi_{K})]) (if one listens at the "correct" output port)
# Thus it is possible to calculate p_R(b|\tilde a)
# The probability of deciding for b is then
#    sum_{phi_1,...,phi_K} p_R(b|\tilde a(phi_1, ..., phi_K))


#import scipy.integrate
import scipy.special as scps
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.special import i1
import scipy
import numpy as np
import random
import itertools
import collections
import math
import mpmath
from scipy.stats import norm
import time


plt.rcParams.update({'font.size': 16}) # change font size in plots

def rX(a, K, kappa, equalPort=True):
    """
    This function calculates the received signal when listening at the right or at the wrong port under iid van Mises noise
    :param a: alpha of the coherent state
    :param K: Order of the Hadamard code (has to be a power of 2!)
    :param kappa: kappa of the van Mises distribution
    :param equalPort: calculates the signal for equal (k' = k) or unequal port (k' =/= k)
    :return: the received signal (Eq. (10) in paper)
    """

    rx = 0
    s = [0 for i in range(K)]
    if kappa != np.inf:
        s = np.random.vonmises(0, kappa, K)

    if equalPort:
        rx = a * sum([np.exp(-complex(0, s[i])) for i in range(K)])/np.sqrt(K)

    else:
        diffSum = sum([np.power(-1,i)*np.exp(complex(0, s[i])) for i in range(int(K))])
        rx = a * diffSum/np.sqrt(K)

    return rx

def rX_norm(a, K, c, equalPort=True):
    """
        This function calculates the received signal when listening at the right or at the wrong port under iid wrapped normal noise
        :param a: alpha of the coherent state
        :param K: Order of the Hadamard code (has to be a power of 2!)
        :param c: sigma of the wrapped normal distribution
        :param equalPort: calculates the signal for equal (k' = k) or unequal port (k' =/= k)
        :return: the received signal (Eq. (10) in paper)
        """
    rx = 0
    s = [0 for i in range(K)]
    if kappa != np.inf:
        s = np.random.normal(0, c, K)

        # Wrap the samples to the interval [0, 2*pi)
        s = np.mod(s, 2 * np.pi)

    if equalPort:
        rx = a * sum([np.exp(-complex(0, s[i])) for i in range(K)]) / np.sqrt(K)
    else:
        diffSum = sum([np.power(-1, i) * np.exp(complex(0, s[i])) for i in range(int(K))])
        rx = a * diffSum / np.sqrt(K)
    return rx


def homodyne(a, displacement, epsilon):
    """
    calculates the output probabilities of the homodyne detector
    :param a: the signal we want to measure
    :param displacement: alpha of the coherent state
    :param epsilon: threshold
    :return: probabilities of homodyne detector (Eq. (28)-(30))
    """

    pMinusA = 0.5*( 1 - scps.erf(np.sqrt(2)*(a+epsilon)))
    pZero = 0.5*( scps.erf(np.sqrt(2)*(epsilon - a)) + scps.erf(np.sqrt(2)*(a+epsilon)))
    pPlusA = 0.5*( 1 - scps.erf(np.sqrt(2)*(epsilon - a)))

    coin = random.random()
    out = displacement
    if coin < pMinusA.real:
        out = -displacement
    elif coin >= pMinusA.real and coin < pMinusA.real + pZero.real:
        out = 0
    elif coin >= pMinusA.real + pZero.real:
        out = displacement

    return out

def sampledQ(e, K, kappa, samples):
    """
    samples the classical channel defined from using a Hadamard receiver with words of length K with BPSK alphabet a=sqrt(e) and b=-sqrt(e)
    :param e: Photon number
    :param K: Order of the Hadamard codewords
    :param kappa: parameter for the von Mises distribution
    :param samples: how many samples we want
    :return: the output has form [q( |a,k=l), q( |b,k=l), q( |a,k#l), q( |b,k#l)]
        ("k = l" indicates the probability distribution at the same port, k#l quantifies the statistics if one listens at the wrong port)
    """

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
    displ = e * np.sqrt(K)
    epsilon = displ / 2

    for i in range(samples):
        # calculate what happens if "+" is sent
        a = displ
        b = -a
        # add noise to a
        rx = rX(a, K, kappa, True)

        # get output of homodyne receiver if listening at the CORRECT port & check if decoded correctly
        out = homodyne(rx, displ, epsilon)

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

def sampledQ_norm(e, K, kappa, samples):
    """
       samples the classical channel defined from using a Hadamard receiver with words of length K with BPSK alphabet a=sqrt(e) and b=-sqrt(e)
       :param e: Photon number
       :param K: Order of the Hadamard codewords
       :param kappa: parameter for the wrapped normal distribution
       :param samples: how many samples we want
       :return: the output has form [q( |a,k=l), q( |b,k=l), q( |a,k#l), q( |b,k#l)]
           ("k = l" indicates the probability distribution at the same port, k#l quantifies the statistics if one listens at the wrong port)
       """
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
    displ = e * np.sqrt(K)
    displ = e * np.sqrt(K)
    epsilon = displ / 2

    for i in range(samples):
        # calculate what happens if "+" is sent
        a = displ
        b = -a
        # add noise to a
        rx = rX_norm(a, K, kappa, True)
        # get output of homodyne receiver if listening at the CORRECT port & check if decoded correctly
        out = homodyne(rx, displ, epsilon)
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
        rx = rX_norm(b, K, kappa, True)
        out = homodyne(rx, displ, epsilon)
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
        rx = rX_norm(a, K, kappa, False)
        out = homodyne(rx, displ, epsilon)
        if out == a:
            a0a += 1
        elif out == b:
            b0a += 1
        elif out == 0:
            zero0a += 1
        rx = rX_norm(b, K, kappa, False)
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
    """
    Calculates the conditional distribution
    :param a: transmitted symbol at port k (0 or 1)
    :param bK:
    :param k: which port
    :param q: output of sampledQ(...)
    :return: the conditional distribution
    """
    resultAtCorrectPort = int(bK[k])
    count = collections.Counter(bK)

    nPlus1 = count[0]

    nMinus1 = count[1]

    nZero = len(bK) - nPlus1 - nMinus1

    if resultAtCorrectPort == 0:
        nPlus1 -= 1
    if resultAtCorrectPort == 1:
        nMinus1 -= 1
    if resultAtCorrectPort == 2:
        nZero -= 1

    # probability of detecting b given a at the RIGHT port is q[0][a][b]
    # probability of detecting b given a at the WRONG port is q[1][a][b]
    plus = [q[1][a][0] for i in range(nPlus1)]
    minus = [q[1][a][1] for i in range(nMinus1)]
    zero = [q[1][a][2] for i in range(nZero)]

    prodPlus = 1
    prodMinus = 1
    prodZero = 1
    if len(plus) > 0:
        prodPlus = q[1][a][0] ** nPlus1
    if len(minus) > 0:
        prodMinus = q[1][a][1] ** nMinus1
    if len(zero) > 0:
        prodZero = q[1][a][2] ** nZero
    out = q[0][a][resultAtCorrectPort] * prodPlus * prodMinus * prodZero
    return out



def outDistrib(q, bK):
    """
    Calculates the output distribution
    :param q: output of sampledQ(...)
    :param bK:
    :return: the output distribution
    """
    K = len(bK)
    oD = (1 / K) * (1 / 2) * sum([condDistrib(a, bK, k, q) for k in range(K) for a in range(2)])
    return oD

def pLogP(p):
    """
    Calculates p * log_2(p)
    :param p: probability
    :return: p * log_2(p)
    """
    out = 0
    if p > 0:
        out = p * np.log2(p)
    return out

def size(measType):
    """
    Alphabet of the measurement ({0,1,2}^K)
    :param measType: alphabet
    :return: size of alphabet
    """
    length = sum([x for x in measType])
    nPlus1 = measType[0]
    nMinus1 = measType[1]
    return scps.binom(length, nPlus1) * scps.binom(length - nPlus1, nMinus1)

def mutualInformation(q, K):
    """
    Calculates the Mutual Information
    :param q: output of sampledQ(...)
    :param K: Order of Hadamard
    :return: the mutual information
    """
    condH = 0
    outH = 0
    if K > 1:
        # we look at the cases where the correct port is - without loss of generality - port 0
        # define how many 1's are detected (logical 0), at ports other than port 0:
        for i in range(K):
            # define how many -1's are detected (logical 1), at ports other than port 1:
            for j in range(K - i):
                # define what was sent at port 0:
                for a in range(2):
                    # define what was received at port 0:
                    for b in range(3):
                        # create string of measurements, without loss of generality well-ordered:
                        resultAtPortK = [b]
                        bK = resultAtPortK + [0 for m in range(i)]
                        bK = bK + [1 for m in range(j)]
                        # define how many 0's (logical 2) are detected:
                        bK = bK + [2 for m in range(K - 1 - i - j)]
                        # assume the correct port (port 0) was chosen by the sender:
                        cd = condDistrib(a, bK, 0, q)
                        if cd > 0:
                            condH -= size([i, j, K - 1 - i - j]) * cd * np.log2(cd)
    # there is a total of 2 phases available for the receiver, so we have to divide by two:
    condH = condH / 2
    # there are a total of 8 ports, each of them behaves identically, and the sender averages uniformly over the ports. Thus condH is calculated now.
    # the output entropy:
    if K > 1:
        # we look at the cases where the correct port is - without loss of generality - port 0
        # define how many 1's are detected (logical 0), at ports other than port 0:
        for i in range(K):
            # define how many -1's are detected (logical 1), at ports other than port 1:
            for j in range(K - i):
                bK = [0 for m in range(i)]
                bK = bK + [1 for m in range(j)]
                # define how many 0's (logical 2) are detected:
                bK = bK + [2 for m in range(K - 1 - i - j)]
                # now average over the input variables:
                for b in range(3):
                    # now bK is defined up to permutation on "zero output ports"
                    od = 0
                    for a in range(2):
                        for k in range(K):
                            cd = condDistrib(a, [b] + bK, k, q)
                            if cd > 0:
                                od += cd
                    od = od / (K * 2)
                    if od > 0:
                        outH -= size([i, j, K - 1 - i - j]) * od * np.log2(od)
    return outH - condH

def capacity(q, K):
    """
    Calculates the capacity
    :param q: output of sampledQ(...)
    :param K: Order of Hadamard
    :return: capacity
    """
    return mutualInformation( q, K)/K

def shannonCapacity( kappa, e, samples=1000 ):
    """
    Calculates the Shannon capacity
    :param kappa: parameter for von Mises distribution
    :param e: Photon number
    :param samples: how many samples we want
    :return: Shannon capacity
    """
    s = samples
    qfull = sampledQ( e, 1, kappa, s )
    q = qfull[0]

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
    """
    calculates kappa for noise model (see paper section V.A)
    :param N: number of photons per second
    :param B: baud rate
    :return: kappa
    """
    n = math.log(N, 10)
    b = math.log(B, 10)
    return (np.power(10, 19 - b) / 6)


def g (x):
    # needed for next function
    return np.log2( 1 + x) + x*np.log2(1 + 1/x)


def sHol (tau, ns, noise):
    """
    Spectral efficiency according to Holevo formula
    :param tau: loss
    :param ns: photon number
    :param noise: noise
    :return: Holevo Spectral Efficency
    """
    if noise>0:
        return g( ns*tau + noise ) - g( noise)
    else:
        return g( ns*tau + noise )

if __name__ == "__main__" :
    # lists for plots
    liste = []
    liste2 = []
    liste_h = []
    liste2_h = []

    listeg = []
    liste2g = []
    liste_hg = []
    liste2_hg = []


    # check mutual info
    mutualInfoCheck = True
    # change to True if plots are wanted
    plot = True

    # check baudrate calucaltions
    baudrateCheck = True

    if mutualInfoCheck:

        start_time = time.time()
        for j in [1, 0.1, 0.001]:

            kap = 1
            a = j
            samples = 1000

            print("a=", a, "kappa=", kap, "samples=", samples)
            m1 = shannonCapacity(kap, a)
            print("benchmark is Shannon capacity", m1)
            liste.append(m1)
            liste2.append(m1)
            for i in range(7):
                K = int(np.power(2, i))
                q = sampledQ(a, K, kap, samples)


                m = mutualInformation(q, K)
                c = m / K
                liste.append(m)
                liste2.append(c)
                print("at K=", K, "we have capacity", m, c)
            for j in [1, 0.1, 0.001]:
                kap = 1
                a = j
                samples = 1000
                print("a=", a, "kappa=", kap, "samples=", samples)
                m1 = shannonCapacity(kap, a)
                print("benchmark is Shannon capacity", m1)
                listeg.append(m1)
                liste2g.append(m1)
                for i in range(7):
                    K = int(np.power(2, i))
                    q = sampledQ_norm(a, K, kap, samples)
                    m = mutualInformation(q, K)
                    c = m / K
                    listeg.append(m)
                    liste2g.append(c)
                    print("at K=", K, "we have capacity", m, c)
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Elapsed time: {elapsed_time} seconds")
        if plot:
            plt.figure()
            xlables = [1, 2, 4, 8, 16, 32, 64]
            samples = 1000
            plt.title("Mutual Information of the Joint Detection Receiver with sigma= " + str(
                kap) + "and kappa =" + str(kap))
            plt.plot(range(1, 7), liste[2:8], label="vM quantum a = 1")
            plt.plot(range(1, 7), liste[10:16], label="vM quantum a = 0.01")
            plt.plot(range(1, 7), liste[18:24], label="vM quantum a = 0.001")
            plt.plot(range(1, 7), listeg[2:8], label="WN quantum a = 1")
            plt.plot(range(1, 7), listeg[10:16], label="WN quantum a = 0.01")
            plt.plot(range(1, 7), listeg[18:24], label="WN quantum a = 0.001")
            plt.legend()
            plt.xlim(1, 7)
            plt.xticks(range(0, 7), xlables)
            plt.xlabel('n', labelpad=3)
            plt.ylabel('Mutual Information')
            plt.grid()

            plt.figure()
            plt.title("Capacity of the Joint Detection Receiver with sigma= " + str(
                kap) + " samples = " + str(samples))
            plt.plot(range(1, 7), liste2[2:8], label="vM quantum a = 1")
            plt.plot(range(1, 7), liste2[10:16], label="vM quantum a = 0.01")
            plt.plot(range(1, 7), liste2[18:24], label="vM quantum a = 0.001")
            plt.plot(range(1, 7), liste2g[2:8], label="N quantum a = 1")
            plt.plot(range(1, 7), liste2g[10:16], label="N quantum a = 0.01")
            plt.plot(range(1, 7), liste2g[18:24], label="N quantum a = 0.001")
            plt.legend()
            plt.xlim(1, 7)
            plt.xticks(range(0, 7), xlables)
            plt.xlabel('n', labelpad=3)
            plt.ylabel('Capacity')
            plt.grid()
            plt.show()

    if baudrateCheck:
        # lists for plots
        baudrate_list = []
        kappa_list = []
        mutInfo_list = []
        shanInfo_list = []
        hadaCap_list = []
        shanCap_list = []
        photnum_list = []
        kap = 0.001
        A = np.exp(-0.05 * 250) * 10 ** 16
        samples = 1000
        print("A=", A, "kappa=", kap, "samples=", samples)
        holData = []
        shData = []
        holVar_list_2 = []
        holVar_list_4 = []
        holVar_list_8 = []
        holVar_list_16 = []
        holVar_list_32 = []
        shVar_list = []
        holavg_list_2 = []
        holavg_list_4 = []
        holavg_list_8 = []
        holavg_list_16 = []
        holavg_list_32 = []
        shavg_list = []

        for b in range(30):
            # loop over baudrates for shannon capacity

            br = (100 + b * 10) * (10 ** 9)
            print("photon number is", A / br)
            baudrate_list.append(br / (10 ** 9))
            kap = kappa(10 ** 16, br)
            print("kap:",kap)

            shAvg = 0
            holCapacities = []
            shCapacities = []
            for step in range(20):
                # get average shannon capacity
                shCap = shannonCapacity(kap, A / br)
                shAvg += shCap
                shCapacities += [shCap]

            shVariance = 0
            shAvg = shAvg / 20
            shavg_list.append(shAvg * br)
            for step in range(20):
                # get shannon capacity variance
                shVariance += (shAvg - shCapacities[step]) ** 2
            shVariance = np.sqrt(shVariance / (20 - 1))
            shVar_list.append(shVariance * br)

            print("at baudrate=", br, "Shannon average=", shAvg, "variance=", shVariance)

        for b in range(30):
            # loop over baudrate for jdr capacity
            br = (100 + b * 10) * (10 ** 9)
            print("photon number is", A / br)

            kap = kappa(10 ** 16, br)

            holAvg = 0
            holCapacities = []
            shCapacities = []
            for k in [4, 32]:
                holCapacities = []
                holAvg = 0
                for step in range(20):
                    K = k
                    q = sampledQ(A / br, K, kap, samples)
                    m = mutualInformation(q, K)

                    holCap = m / K
                    holAvg += holCap
                    holCapacities += [holCap]

                holVariance = 0
                holAvg = holAvg / (20)

                if K == 4:
                    holavg_list_4.append(holAvg * br)

                if K == 32:
                    holavg_list_32.append(holAvg * br)

                for step2 in range(20):
                    holVariance += (holAvg - holCapacities[step2]) ** 2

                holVariance = np.sqrt(holVariance / (20 - 1))

                if K == 4:
                    holVar_list_4.append(holVariance * br)

                if K == 32:
                    holVar_list_32.append(holVariance * br)

                print("at baudrate=", br, "Holevo average=", holAvg, "variance=", holVariance, "K= ", K)

        if plot:
            blah = 30
            xlables = baudrate_list
            plt.figure()
            clrs = sns.color_palette("husl", 5)
            plt.title("Holevo Average with Variance with lowest expected photon number at receiver is " + str(
                A / br))
            var4min = []
            var4plus = []
            for i in range(len(holVar_list_4)):
                var4min.append(holavg_list_4[i] - holVar_list_4[i])
                var4plus.append(holavg_list_4[i] + holVar_list_4[i])
            var32min = []
            var32plus = []
            for i in range(len(holVar_list_4)):
                var32min.append(holavg_list_32[i] - holVar_list_32[i])
                var32plus.append(holavg_list_32[i] + holVar_list_32[i])
            varshmin = []
            varshplus = []
            for i in range(len(holVar_list_4)):
                varshmin.append(shavg_list[i] - shVar_list[i])
                varshplus.append(shavg_list[i] + shVar_list[i])

            with sns.axes_style("darkgrid"):

                plt.plot(range(blah), holavg_list_4, c=clrs[0])
                plt.fill_between(range(blah), var4min, var4plus, alpha=0.3, facecolor=clrs[0])

                plt.plot(range(blah), holavg_list_32, c=clrs[1])
                plt.fill_between(range(blah), var32min, var32plus, alpha=0.3,
                                 facecolor=clrs[1])

                plt.xticks(range(0, blah), xlables)
                plt.xlabel('Br * 10^9')
                plt.ylabel('Capacity')
                plt.grid()
                plt.plot(range(blah), shavg_list, c=clrs[2])
                plt.fill_between(range(blah), varshmin, varshplus, alpha=0.3, facecolor=clrs[2])  # label = "classical")
                plt.xticks(range(0, blah), xlables)
                plt.xlabel('Br * 10^9')
                plt.ylabel('Capacity')
                plt.tight_layout()
                plt.xticks(rotation=90)

                plt.show()


