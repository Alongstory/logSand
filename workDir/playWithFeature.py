
if __name__ == '__main__':
    featureDic = {}
    with open ('featureDict', 'r') as f:
        for i, l in enumerate(f):
            pass
        i = i + 1
        #counter
    with open ('featureDict', 'r') as f:
        for line in f:
            a = eval(line)
            b = [0] * i
            b[a[1]] = 1
            featureDic[a[0]] = b
            print 'avc'
    f.close()
    print repr(featureDic)

    with open ('dictionary' , 'w') as f1:
        f1.write(repr(featureDic))


