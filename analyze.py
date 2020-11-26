'''
    EF513 Music Project (1)
    gihanjayatilaka[at]eng[dot]pdn[dot]ac[dot]lk
    
    TASK: Write a computer program to extract the fundamental frequency and the harmonic partials of wave forms that belong to the following instruments Guitar, Flute, Violin.
'''
#Global variables (Not needed in python, but for convinience in reading)
DEBUG=True
OUTPUT_DIR=None
log=None

import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt


def initializeLog(fileName):
    f=open(fileName,"w")
    f.write("STARTING LOG....")
    return f

def describeArray(caption,ar):
    log.write("DESCRIBING : {}\n".format(caption))
    # print("{:.2f}".format(0))
    log.write("Size={} | Min={:.2f} | Max={:.2f} | Avg={:.2f} | Std={:.2f}\n".format(ar.shape,np.min(ar),np.max(ar),np.mean(ar),np.std(ar)))


def nonZeroPartials(xx,ff,N=10,freqWindowHalf=50):
    ans=findHarmonicPartials(xx,ff,N,freqWindowHalf)
    ans=refineHarmonicPartials(ans["f0"],ans["fn"],ans["An"])
    return ans




def refineHarmonicPartials(f0,partialsF,partialsA):
    log.write("CALLING : refineHarmonicPartials({},{},{})\n".format(f0,partialsF,partialsA))
    TOLERANCE_F=0.2
    THERSHOLD_A=5
    
    newPartialMultiplicant=[]
    newPartialF=[]
    newPartialA=[]
    
    for multiplicant in range(1,10):
        # multiplicant=2**exponent
        fn=f0*multiplicant

        freqDiff=np.abs(partialsF - fn)
        fIdx=np.argmin(freqDiff)

        if freqDiff[fIdx]/f0 < TOLERANCE_F and partialsA[fIdx]>THERSHOLD_A:
            newPartialF.append(partialsF[fIdx])
            newPartialA.append(partialsA[fIdx])
            newPartialMultiplicant.append(multiplicant)
    ans={"f0":f0,"fn":newPartialF,"An":newPartialA,"n":newPartialMultiplicant}
    log.write("RETURNING : refineHarmonicPartials() = ans\n")
    return ans








def findHarmonicPartials(xx,ff,N=10,freqWindowHalf=50):
    log.write("CALLING : findHarmonicPartials({},{},{},{})\n".format(xx,ff,N,freqWindowHalf))
    partials= findPartials(xx,ff,N,freqWindowHalf)
    partialF=partials["freq"]
    partialA=partials["amp"]
    error=np.zeros((6),dtype=np.float32)
    for n0 in range(6):
        multiplicantInt=(partialF/partialF[n0]).astype(np.int32)
        multiplicantFloat=partialF/partialF[n0]
        error[n0]=np.sum(partialA * np.abs(multiplicantFloat-multiplicantInt))

    log.write("Calculated errors : {}".format(error))

    n0 = np.argmin(error)
    f0 = partialF[n0]

    ans={"f0":f0, "fn":partialF, "An":partialA}
    log.write("RETURNING : findHarmonicPartials() = {}\n".format(ans))
    return ans



def findPartials(xx,ff,N,freqWindowHalf):
    log.write("findPartials({},{},{},{})\n".format(xx,ff,N,freqWindowHalf))
    x=np.array(xx)
    f=np.array(ff)

    xMax=np.max(x)

    maxF=[]
    maxX=[]

    for n in range(N):
        fn = np.argmax(x)
        maxX.append(x[fn])
        maxF.append(f[fn])
        plt.figure()
        plt.plot(f,x,".b",maxF[-1],maxX[-1],"*r")
        plt.ylim(0,xMax*1.2)
        plt.title("Iter = {}".format(n))
        plt.savefig("{}/partial-finding-{:02}.png".format(OUTPUT_DIR,n))
        log.write("Ssaved plot while removing partials : {}\n".format("{}/partial-finding-{:02}.png".format(OUTPUT_DIR,n)))
        x[max(0,fn-freqWindowHalf):min(fn+freqWindowHalf,x.shape[0]-1)]=0
    

    maxF=np.array(maxF)
    maxX=np.array(maxX)

    order=np.argsort(maxF)
    maxF=maxF[order]
    maxX=maxX[order]
    ans={"freq":maxF,"amp":maxX}

    log.write("RETURNING: findPartials() == {}\n".format(ans))
    return ans

def writeToFile(f0,n,fn,An,fileName):
    fileOut=open(fileName,"w")
    fileOut.write("Fundamental-frequency : {:02}\n".format(f0))
    fileOut.write("n\tfn\tAn\n")
    for i in range(len(fn)):
        fileOut.write("{}\t{:.2f}\t{:.2f}\n".format(n[i],fn[i],An[i]))
    fileOut.close()
    print("Finished writing to {}".format(fileName))


if __name__=="__main__":
    

    plt.rcParams.update({'figure.max_open_warning': 0})


    args=argparse.ArgumentParser()
    args.add_argument("--inputFile","-i",dest="inputFile",type=str)
    args.add_argument("--outputDir","-o",dest="outputDir",type=str)

    args=args.parse_args()

    if args.outputDir[-1]=='/':
        args.outputDir=args.outputDir[:-1]
    OUTPUT_DIR=args.outputDir

    log=initializeLog("{}/log.txt".format(OUTPUT_DIR))
    log.write("Starting program with arguments {}\n".format(args))

    log.write("INPUT FILE: {}\n".format(args.inputFile))
    log.write("OUTPUT DIR: {}\n".format(args.outputDir))

    x , sr = librosa.load(args.inputFile)
    t = np.arange(x.shape[0])*(1.0/sr)

    print("Read file {} with sampling rate = {}".format(args.inputFile,sr))
    log.write(("Read file {} with sampling rate = {}\n".format(args.inputFile,sr)))

    describeArray("x",x)
    describeArray("t",t)

    plt.plot(t,x)
    plt.savefig("{}/timeSeries.png".format(OUTPUT_DIR))
    log.write("Saved fig {}\n".format("{}/timeSeries.png".format(OUTPUT_DIR)))

    fourierComplex=np.fft.fft(x)
    fourierAmplitude=np.sqrt(np.power(fourierComplex.real ,2)+np.power(fourierComplex.imag ,2))
    fourierFrequency=np.fft.fftfreq(x.shape[0],1.0/sr)

    fourierAmplitude=fourierAmplitude[:int(fourierAmplitude.shape[0]/2)]
    fourierFrequency=fourierFrequency[:int(fourierFrequency.shape[0]/2)]

    describeArray("fourierComplex",fourierComplex)
    describeArray("fourierAmplitude",fourierAmplitude)
    describeArray("fourierFrequency",fourierFrequency)


    plt.plot(fourierFrequency,fourierAmplitude)
    plt.savefig("{}/freqSpectrum.png".format(OUTPUT_DIR))
    log.write("Saved fig {}\n".format("{}/freqSpectrum.png".format(OUTPUT_DIR)))

    if str(args.inputFile).count("guitar") > 0:
        print("Guitar config")
        partials =nonZeroPartials(fourierAmplitude,fourierFrequency,N=15,freqWindowHalf=50)
    else:
        print("Violin and flute config")
        partials =nonZeroPartials(fourierAmplitude,fourierFrequency,N=10,freqWindowHalf=250)
    

    fig=plt.figure()
    plt.plot(fourierFrequency,fourierAmplitude,"b",partials["fn"],partials["An"],"r.")
    plt.savefig("{}/harmonics.png".format(OUTPUT_DIR))


    writeToFile(partials["f0"],partials["n"],partials["fn"],partials["An"],"{}/output.txt".format(OUTPUT_DIR))

    log.write("Finished running inputs {}\n".format(args))
    print("Finished running inputs {}".format(args))


