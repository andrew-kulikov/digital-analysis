def main():
    x, y, z = map(int, input().split())
    g = list(map(int, input().split()))
    d = list(map(int, input().split()))
    h = list(map(int, input().split()))
    x = []
    y = []
    for i in range(100000):
        x.append(0)
        y.append(0)
    for i in d:
        x[i] += 1
    for i in h:
        y[i] += 1
    for i in range(100000 - 1, 0, -1):
        x[i - 1] += x[i]
    
    

if __name__ == '__main__':
    main()
