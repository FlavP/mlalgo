def is_prime(n):
    for i in range(2,n-1):
        if n % i == 0:
            #print(i)
            return False
    return True

def primes(n):
    list_of = []
    #2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97
    for i in range(2, n):
        if is_prime(i):
            #print(i)
            list_of.append(i)
    return list_of
test_sample = primes(100)
print(test_sample)